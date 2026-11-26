"""Async gRPC Coordinator for two-stage pipeline-parallel LLM inference."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Any

import grpc
import torch

from common.config import load_config
from common.serialization import bytes_to_tensor
from coordinator.metrics import MetricsCollector, RequestMetrics, StepMetrics
from coordinator.request_manager import (
    OverloadError,
    RequestManager,
    RequestRecord,
    RequestState,
    StaleResponseError,
)
from proto.generated import inference_pb2, inference_pb2_grpc

_RETRIABLE_GRPC_CODES = {
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}

_RETRIABLE_APP_STATUSES = {
    inference_pb2.STATUS_STALE_EPOCH,
    inference_pb2.STATUS_OVERLOADED,
    inference_pb2.STATUS_CACHE_MISMATCH,
    inference_pb2.STATUS_ERROR,
}


def _mode_to_proto(mode: str) -> int:
    if mode == "coordinator":
        return inference_pb2.MODE_COORDINATOR_LOGITS
    if mode == "worker":
        return inference_pb2.MODE_WORKER_TOKEN_ID
    raise ValueError(f"Unsupported token selection mode: '{mode}'")


class PipelineStepError(RuntimeError):
    """Raised internally when a pipeline RPC step fails and may trigger replay."""

    def __init__(self, message: str, retriable: bool = True, needs_epoch_sync: bool = False):
        super().__init__(message)
        self.retriable = retriable
        self.needs_epoch_sync = needs_epoch_sync


class Coordinator:
    """Manages tokenization, request scheduling, token commits, metrics, and recovery."""

    def __init__(
        self,
        config: dict | None = None,
        config_path: str | Path = "config.yaml",
        worker1_stub: Any | None = None,
        worker2_stub: Any | None = None,
        tokenizer: Any | None = None,
        retry_backoff_s: float = 0.05,
    ):
        self.config = config if config is not None else load_config(config_path)
        model_cfg = self.config["model"]
        net_cfg = self.config["network"]
        sched_cfg = self.config["scheduling"]
        rel_cfg = self.config["reliability"]

        self.model_name: str = model_cfg["name"]
        self.default_max_new_tokens: int = int(model_cfg.get("max_new_tokens", 32))
        self.default_selection_mode: str = net_cfg.get("token_selection_mode", "coordinator")
        self.worker1_address = f"{net_cfg['worker1_host']}:{net_cfg['worker1_port']}"
        self.worker2_address = f"{net_cfg['worker2_host']}:{net_cfg['worker2_port']}"

        self.rpc_deadline_s: float = float(rel_cfg.get("rpc_deadline_ms", 5000)) / 1000.0
        self.max_retries: int = int(rel_cfg.get("max_retries", 3))
        self.retry_backoff_s: float = retry_backoff_s

        self.request_manager = RequestManager(
            max_concurrent_requests=int(sched_cfg.get("max_concurrent_requests", 8))
        )
        self.metrics = MetricsCollector()

        self._tokenizer = tokenizer
        self._w1_channel: grpc.aio.Channel | None = None
        self._w2_channel: grpc.aio.Channel | None = None
        self._w1_stub = worker1_stub
        self._w2_stub = worker2_stub
        self._owns_channels = worker1_stub is None and worker2_stub is None

        self.worker1_epoch: int = 1
        self.worker2_epoch: int = 1

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def _ensure_stubs(self) -> None:
        if self._w1_stub is None:
            self._w1_channel = grpc.aio.insecure_channel(self.worker1_address)
            self._w1_stub = inference_pb2_grpc.Worker1ServiceStub(self._w1_channel)
        if self._w2_stub is None:
            self._w2_channel = grpc.aio.insecure_channel(self.worker2_address)
            self._w2_stub = inference_pb2_grpc.Worker2ServiceStub(self._w2_channel)

    async def close(self) -> None:
        if self._owns_channels:
            if self._w1_channel is not None:
                await self._w1_channel.close()
                self._w1_channel = None
            if self._w2_channel is not None:
                await self._w2_channel.close()
                self._w2_channel = None

    async def __aenter__(self) -> Coordinator:
        self._ensure_stubs()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def check_health(self) -> dict[str, dict]:
        """Query both workers and synchronize the coordinator's active worker epochs."""
        self._ensure_stubs()
        req = inference_pb2.HealthRequest()
        h1, h2 = await asyncio.gather(
            self._w1_stub.HealthCheck(req, timeout=self.rpc_deadline_s),
            self._w2_stub.HealthCheck(req, timeout=self.rpc_deadline_s),
        )
        if h1.worker_epoch > 0:
            self.worker1_epoch = h1.worker_epoch
        if h2.worker_epoch > 0:
            self.worker2_epoch = h2.worker_epoch
        return {
            "worker1": {
                "ready": h1.ready,
                "worker_epoch": h1.worker_epoch,
                "active_cache_entries": h1.active_cache_entries,
            },
            "worker2": {
                "ready": h2.ready,
                "worker_epoch": h2.worker_epoch,
                "active_cache_entries": h2.active_cache_entries,
            },
        }

    async def reset_caches(self, request_id: str = "", attempt_id: str = "") -> tuple[int, int]:
        """Reset KV-cache state on both workers concurrently."""
        self._ensure_stubs()
        req1 = inference_pb2.ResetCacheRequest(
            request_id=request_id,
            attempt_id=attempt_id,
            worker_epoch=self.worker1_epoch,
        )
        req2 = inference_pb2.ResetCacheRequest(
            request_id=request_id,
            attempt_id=attempt_id,
            worker_epoch=self.worker2_epoch,
        )
        r1, r2 = await asyncio.gather(
            self._w1_stub.ResetCache(req1, timeout=self.rpc_deadline_s),
            self._w2_stub.ResetCache(req2, timeout=self.rpc_deadline_s),
        )
        return int(r1.cleared_entries), int(r2.cleared_entries)

    async def _send_pipeline_step(
        self,
        record: RequestRecord,
        step_id: int,
        input_ids: list[int],
        cache_position: int,
        is_prefill: bool,
        is_replay: bool = False,
    ) -> tuple[int | None, torch.Tensor | None, StepMetrics]:
        """Execute a single RunPipelineStep RPC and validate response metadata."""
        self._ensure_stubs()
        expected_w1_epoch = self.worker1_epoch
        expected_w2_epoch = self.worker2_epoch

        rpc_req = inference_pb2.Stage1Request(
            request_id=record.request_id,
            attempt_id=record.attempt_id,
            step_id=step_id,
            worker1_epoch=expected_w1_epoch,
            worker2_epoch=expected_w2_epoch,
            input_ids=input_ids,
            cache_position=cache_position,
            is_prefill=is_prefill,
            selection_mode=_mode_to_proto(record.selection_mode),
        )

        t0 = time.perf_counter()
        try:
            resp = await self._w1_stub.RunPipelineStep(rpc_req, timeout=self.rpc_deadline_s)
        except grpc.aio.AioRpcError as exc:
            retriable = exc.code() in _RETRIABLE_GRPC_CODES
            raise PipelineStepError(
                f"gRPC error on step {step_id}: {exc.code().name} ({exc.details()})",
                retriable=retriable,
                needs_epoch_sync=True,
            ) from exc
        step_latency_ms = (time.perf_counter() - t0) * 1000.0

        # Validate application-level status
        if resp.status != inference_pb2.STATUS_OK:
            retriable = resp.status in _RETRIABLE_APP_STATUSES
            needs_sync = resp.status == inference_pb2.STATUS_STALE_EPOCH
            raise PipelineStepError(
                f"Worker returned status={resp.status}: {resp.error_message}",
                retriable=retriable,
                needs_epoch_sync=needs_sync,
            )

        # Fence stale responses by request_id, attempt_id, step_id, and both worker epochs
        if (
            resp.request_id != record.request_id
            or resp.attempt_id != record.attempt_id
            or resp.step_id != step_id
        ):
            raise StaleResponseError(
                f"Mismatched response metadata: got ({resp.request_id}, {resp.attempt_id}, "
                f"{resp.step_id}), expected ({record.request_id}, {record.attempt_id}, {step_id})"
            )
        if resp.worker1_epoch != expected_w1_epoch or resp.worker2_epoch != expected_w2_epoch:
            raise PipelineStepError(
                f"Stale worker epochs in response: got ({resp.worker1_epoch}, "
                f"{resp.worker2_epoch}), expected ({expected_w1_epoch}, {expected_w2_epoch})",
                retriable=True,
                needs_epoch_sync=True,
            )

        step_metrics = StepMetrics(
            request_id=record.request_id,
            attempt_id=record.attempt_id,
            step_id=step_id,
            is_prefill=is_prefill,
            is_replay=is_replay,
            step_latency_ms=step_latency_ms,
            stage1_compute_ms=float(resp.stage1_compute_ms),
            stage1_queue_wait_ms=float(resp.stage1_queue_wait_ms),
            stage2_compute_ms=float(resp.stage2_compute_ms),
            stage2_queue_wait_ms=float(resp.stage2_queue_wait_ms),
            activation_payload_bytes=int(resp.activation_payload_bytes),
            rpc_control_bytes=int(resp.rpc_control_bytes) + rpc_req.ByteSize(),
            logits_payload_bytes=len(resp.logits_data),
        )

        if is_replay:
            # Suppress replay output; cache has been reconstructed up to len(input_ids).
            return None, None, step_metrics

        logits_tensor: torch.Tensor | None = None
        if record.selection_mode == "coordinator":
            logits_tensor = bytes_to_tensor(
                resp.logits_data, list(resp.tensor_shape), resp.dtype
            )
            token_id = int(logits_tensor.argmax(dim=-1).item())
        else:
            token_id = int(resp.selected_token_id)

        return token_id, logits_tensor, step_metrics

    async def _recover_and_replay(
        self,
        record: RequestRecord,
        step_metrics_list: list[StepMetrics],
    ) -> float:
        """Sync epochs, reset caches on both workers, and replay committed history."""
        rec_start = time.perf_counter()
        old_attempt_id = record.attempt_id
        self.request_manager.begin_replay(record.request_id, max_retries=self.max_retries)
        if record.state == RequestState.FAILED:
            return time.perf_counter() - rec_start

        if self.retry_backoff_s > 0:
            await asyncio.sleep(self.retry_backoff_s)

        # Refresh worker epochs (tolerating transient unavailability on restart)
        try:
            await self.check_health()
        except grpc.aio.AioRpcError as exc:
            raise PipelineStepError(
                f"Health check failed during recovery: {exc.code().name}",
                retriable=True,
                needs_epoch_sync=True,
            ) from exc

        # Reset affected KV caches on both workers for consistent state
        try:
            await self.reset_caches(request_id=record.request_id, attempt_id=old_attempt_id)
        except grpc.aio.AioRpcError as exc:
            raise PipelineStepError(
                f"Cache reset failed during recovery: {exc.code().name}",
                retriable=True,
                needs_epoch_sync=True,
            ) from exc

        # If tokens were already committed, replay prompt + committed[:-1] and suppress output.
        # This rebuilds both KV caches to length len(prompt_ids) + len(committed_tokens) - 1,
        # so the next decode step resumes seamlessly with committed_tokens[-1].
        if record.committed_tokens:
            replay_ids = record.prompt_ids + record.committed_tokens[:-1]
            _, _, replay_metrics = await self._send_pipeline_step(
                record=record,
                step_id=-1,
                input_ids=replay_ids,
                cache_position=0,
                is_prefill=True,
                is_replay=True,
            )
            step_metrics_list.append(replay_metrics)

        self.request_manager.finish_replay(record.request_id)
        return time.perf_counter() - rec_start

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        stop_on_eos: bool = True,
        capture_logits: bool = False,
        selection_mode: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Run end-to-end distributed greedy decoding for a single prompt."""
        if not prompt:
            raise ValueError("Prompt cannot be empty")
        max_tokens = (
            self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        )
        if max_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_tokens}")

        mode = selection_mode or self.default_selection_mode
        encoded = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        if encoded.shape[1] == 0:
            raise ValueError("Prompt produced 0 tokens after tokenization")
        prompt_ids = encoded[0].tolist()

        record = self.request_manager.create_request(
            prompt=prompt,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            stop_on_eos=stop_on_eos,
            selection_mode=mode,
            capture_logits=capture_logits,
            request_id=request_id,
        )

        if max_tokens == 0:
            req_metrics = RequestMetrics(
                request_id=record.request_id,
                prompt_tokens=len(prompt_ids),
                generated_tokens=0,
                ttft_s=0.0,
                prefill_time_s=0.0,
                total_time_s=0.0,
                status=record.state.value,
            )
            self.metrics.record_request(req_metrics)
            return self._build_result(record, req_metrics)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        step_metrics_list: list[StepMetrics] = []
        decode_latencies_s: list[float] = []
        prefill_time_s = 0.0
        ttft_s = 0.0
        recovery_time_s = 0.0

        start_t = time.perf_counter()

        try:
            while not record.is_terminal:
                is_prefill = record.next_step_id == 0
                step_input = (
                    record.prompt_ids if is_prefill else [record.committed_tokens[-1]]
                )
                cache_pos = record.cache_position
                step_t0 = time.perf_counter()

                try:
                    token_id, logits, s_metrics = await self._send_pipeline_step(
                        record=record,
                        step_id=record.next_step_id,
                        input_ids=step_input,
                        cache_position=cache_pos,
                        is_prefill=is_prefill,
                        is_replay=False,
                    )
                    step_metrics_list.append(s_metrics)
                    assert token_id is not None

                    self.request_manager.commit_step(
                        request_id=record.request_id,
                        attempt_id=record.attempt_id,
                        step_id=record.next_step_id,
                        token_id=token_id,
                        eos_token_id=eos_id,
                        logits=logits,
                    )
                    step_elapsed = time.perf_counter() - step_t0
                    if is_prefill:
                        prefill_time_s = step_elapsed
                        ttft_s = time.perf_counter() - start_t
                    else:
                        decode_latencies_s.append(step_elapsed)

                except (PipelineStepError, StaleResponseError) as exc:
                    retriable = isinstance(exc, StaleResponseError) or exc.retriable
                    if not retriable:
                        self.request_manager.mark_failed(record.request_id, str(exc))
                        break
                    # Run bounded recovery + replay loop
                    while not record.is_terminal:
                        try:
                            recovery_time_s += await self._recover_and_replay(
                                record, step_metrics_list
                            )
                            break
                        except (PipelineStepError, StaleResponseError) as rec_exc:
                            if record.retry_count >= self.max_retries:
                                self.request_manager.mark_failed(
                                    record.request_id,
                                    f"Recovery failed after {record.retry_count} retries: {rec_exc}",
                                )
                                break
        finally:
            # Clean up worker caches once the request reaches a terminal state
            if record.is_terminal:
                try:
                    await self.reset_caches(request_id=record.request_id, attempt_id="")
                except Exception:
                    pass

        total_time_s = time.perf_counter() - start_t
        req_metrics = RequestMetrics(
            request_id=record.request_id,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(record.committed_tokens),
            ttft_s=ttft_s,
            prefill_time_s=prefill_time_s,
            decode_token_latencies_s=decode_latencies_s,
            total_time_s=total_time_s,
            total_activation_bytes=sum(s.activation_payload_bytes for s in step_metrics_list),
            total_rpc_control_bytes=sum(s.rpc_control_bytes for s in step_metrics_list),
            total_logits_bytes=sum(s.logits_payload_bytes for s in step_metrics_list),
            retry_count=record.retry_count,
            recovery_time_s=recovery_time_s,
            status=record.state.value,
            step_metrics=step_metrics_list,
        )
        self.metrics.record_request(req_metrics)
        return self._build_result(record, req_metrics)

    async def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int | None = None,
        stop_on_eos: bool = True,
        capture_logits: bool = False,
        selection_mode: str | None = None,
    ) -> list[dict]:
        """Submit multiple prompts concurrently so requests interleave across stages."""
        tasks = [
            asyncio.create_task(
                self.generate(
                    prompt=p,
                    max_new_tokens=max_new_tokens,
                    stop_on_eos=stop_on_eos,
                    capture_logits=capture_logits,
                    selection_mode=selection_mode,
                )
            )
            for p in prompts
        ]
        return list(await asyncio.gather(*tasks))

    def _build_result(self, record: RequestRecord, req_metrics: RequestMetrics) -> dict:
        return {
            "request_id": record.request_id,
            "status": record.state.value,
            "error_message": record.error_message,
            "prompt_ids": record.prompt_ids,
            "new_token_ids": record.committed_tokens,
            "generated_ids": record.generated_ids,
            "generated_text": self.tokenizer.decode(record.generated_ids),
            "logits_history": record.logits_history,
            "prefill_time_s": req_metrics.prefill_time_s,
            "decode_token_latencies_s": req_metrics.decode_token_latencies_s,
            "ttft_s": req_metrics.ttft_s,
            "total_time_s": req_metrics.total_time_s,
            "retry_count": req_metrics.retry_count,
            "recovery_time_s": req_metrics.recovery_time_s,
            "total_activation_bytes": req_metrics.total_activation_bytes,
            "total_rpc_control_bytes": req_metrics.total_rpc_control_bytes,
            "total_logits_bytes": req_metrics.total_logits_bytes,
        }
