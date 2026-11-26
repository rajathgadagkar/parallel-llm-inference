"""Unit and in-process parity tests for Coordinator, RequestManager, and MetricsCollector."""

from __future__ import annotations

import asyncio
import pytest
import torch

from common.config import load_config
from common.serialization import bytes_to_tensor, tensor_to_bytes
from coordinator.coordinator import Coordinator
from coordinator.metrics import MetricsCollector
from coordinator.request_manager import OverloadError, RequestManager, RequestState, StaleResponseError
from model.baseline import BaselineInference
from model.partition import Stage1Partition, Stage2Partition
from proto.generated import inference_pb2


class InProcessPipelineStubs:
    """Wraps real Stage1Partition + Stage2Partition behind async gRPC-like stubs.

    Supports fault injection (epoch bump / simulated crash on a target step) so we can
    verify end-to-end Coordinator correctness, concurrency, and replay without sockets.
    """

    def __init__(self, s1: Stage1Partition, s2: Stage2Partition):
        self.s1 = s1
        self.s2 = s2
        self.w1_epoch = 1
        self.w2_epoch = 1
        self.fail_on_calls: dict[tuple[str, int], str] = {}  # (request_id, step_id) -> failure_type
        self.reset_calls: list[tuple[str, str, str]] = []  # (worker, req_id, att_id)

    @property
    def worker1_stub(self):
        return _W1Stub(self)

    @property
    def worker2_stub(self):
        return _W2Stub(self)


class _W1Stub:
    def __init__(self, backend: InProcessPipelineStubs):
        self._b = backend

    async def HealthCheck(self, req, timeout=None):
        return inference_pb2.HealthResponse(
            ready=True,
            worker_epoch=self._b.w1_epoch,
            active_cache_entries=self._b.s1.active_entries(),
        )

    async def ResetCache(self, req, timeout=None):
        self._b.reset_calls.append(("w1", req.request_id, req.attempt_id))
        if not req.request_id:
            n = self._b.s1.active_entries()
            self._b.s1.clear()
        else:
            n = self._b.s1.reset(req.request_id, req.attempt_id or None)
        return inference_pb2.ResetCacheResponse(
            status=inference_pb2.STATUS_OK,
            cleared_entries=n,
        )

    async def RunPipelineStep(self, req, timeout=None):
        # Yield to event loop so concurrent requests genuinely interleave
        await asyncio.sleep(0)

        key = (req.request_id, req.step_id)
        if key in self._b.fail_on_calls:
            failure = self._b.fail_on_calls.pop(key)
            if failure == "w2_crash":
                # Simulate Worker 2 crashing and restarting with a bumped epoch & wiped cache
                self._b.w2_epoch += 1
                self._b.s2.clear()
                return inference_pb2.Stage1Response(
                    request_id=req.request_id,
                    attempt_id=req.attempt_id,
                    step_id=req.step_id,
                    worker1_epoch=self._b.w1_epoch,
                    worker2_epoch=self._b.w2_epoch,
                    status=inference_pb2.STATUS_STALE_EPOCH,
                    error_message="Worker 2 restarted",
                )
            if failure == "always_overloaded":
                # Keep failing on every retry
                self._b.fail_on_calls[key] = "always_overloaded"
                return inference_pb2.Stage1Response(
                    request_id=req.request_id,
                    attempt_id=req.attempt_id,
                    step_id=req.step_id,
                    worker1_epoch=self._b.w1_epoch,
                    worker2_epoch=self._b.w2_epoch,
                    status=inference_pb2.STATUS_OVERLOADED,
                    error_message="Queue full",
                )

        if req.worker1_epoch != self._b.w1_epoch or req.worker2_epoch != self._b.w2_epoch:
            return inference_pb2.Stage1Response(
                request_id=req.request_id,
                attempt_id=req.attempt_id,
                step_id=req.step_id,
                worker1_epoch=self._b.w1_epoch,
                worker2_epoch=self._b.w2_epoch,
                status=inference_pb2.STATUS_STALE_EPOCH,
                error_message="Epoch mismatch",
            )

        inp = torch.tensor([list(req.input_ids)], dtype=torch.long)
        h = self._b.s1.forward(
            req.request_id,
            req.attempt_id,
            inp,
            expected_cache_pos=req.cache_position,
        )
        raw_act, act_shape, act_dtype = tensor_to_bytes(h)
        h_recv = bytes_to_tensor(raw_act, act_shape, act_dtype)

        mode = (
            "worker"
            if req.selection_mode == inference_pb2.MODE_WORKER_TOKEN_ID
            else "coordinator"
        )
        out = self._b.s2.forward(
            req.request_id,
            req.attempt_id,
            h_recv,
            expected_cache_pos=req.cache_position,
            selection_mode=mode,
        )

        if mode == "worker":
            return inference_pb2.Stage1Response(
                request_id=req.request_id,
                attempt_id=req.attempt_id,
                step_id=req.step_id,
                worker1_epoch=self._b.w1_epoch,
                worker2_epoch=self._b.w2_epoch,
                selected_token_id=int(out),
                status=inference_pb2.STATUS_OK,
                activation_payload_bytes=len(raw_act),
                rpc_control_bytes=64,
                stage1_compute_ms=1.0,
                stage1_queue_wait_ms=0.1,
                stage2_compute_ms=1.2,
                stage2_queue_wait_ms=0.1,
            )

        raw_logits, logits_shape, logits_dtype = tensor_to_bytes(out)
        return inference_pb2.Stage1Response(
            request_id=req.request_id,
            attempt_id=req.attempt_id,
            step_id=req.step_id,
            worker1_epoch=self._b.w1_epoch,
            worker2_epoch=self._b.w2_epoch,
            tensor_shape=logits_shape,
            dtype=logits_dtype,
            logits_data=raw_logits,
            status=inference_pb2.STATUS_OK,
            activation_payload_bytes=len(raw_act),
            rpc_control_bytes=64,
            stage1_compute_ms=1.0,
            stage1_queue_wait_ms=0.1,
            stage2_compute_ms=1.2,
            stage2_queue_wait_ms=0.1,
        )


class _W2Stub:
    def __init__(self, backend: InProcessPipelineStubs):
        self._b = backend

    async def HealthCheck(self, req, timeout=None):
        return inference_pb2.HealthResponse(
            ready=True,
            worker_epoch=self._b.w2_epoch,
            active_cache_entries=self._b.s2.active_entries(),
        )

    async def ResetCache(self, req, timeout=None):
        self._b.reset_calls.append(("w2", req.request_id, req.attempt_id))
        if not req.request_id:
            n = self._b.s2.active_entries()
            self._b.s2.clear()
        else:
            n = self._b.s2.reset(req.request_id, req.attempt_id or None)
        return inference_pb2.ResetCacheResponse(
            status=inference_pb2.STATUS_OK,
            cleared_entries=n,
        )


@pytest.fixture(scope="module")
def cfg():
    return load_config("config.yaml")


@pytest.fixture(scope="module")
def baseline(cfg):
    return BaselineInference(model_name=cfg["model"]["name"], dtype=cfg["model"]["dtype"])


@pytest.fixture(scope="module")
def partitions(cfg):
    s1 = Stage1Partition(cfg["model"]["name"], split_layer=3, dtype=cfg["model"]["dtype"])
    s2 = Stage2Partition(cfg["model"]["name"], split_layer=3, dtype=cfg["model"]["dtype"])
    return s1, s2


def test_request_manager_admission_and_ordering():
    rm = RequestManager(max_concurrent_requests=2)
    r1 = rm.create_request("p1", [1, 2], max_new_tokens=2)
    rm.create_request("p2", [3, 4], max_new_tokens=2)
    with pytest.raises(OverloadError):
        rm.create_request("p3", [5, 6], max_new_tokens=2)

    with pytest.raises(StaleResponseError):
        rm.commit_step(r1.request_id, "att-999", 0, 10)
    with pytest.raises(StaleResponseError):
        rm.commit_step(r1.request_id, r1.attempt_id, 1, 10)

    rm.commit_step(r1.request_id, r1.attempt_id, 0, 10)
    assert r1.state == RequestState.ACTIVE
    rm.commit_step(r1.request_id, r1.attempt_id, 1, 11)
    assert r1.state == RequestState.COMPLETED
    assert rm.active_count == 1


@pytest.mark.parametrize("selection_mode", ["coordinator", "worker"])
def test_coordinator_matches_baseline_mode_a_and_b(cfg, baseline, partitions, selection_mode):
    s1, s2 = partitions
    s1.clear()
    s2.clear()
    backend = InProcessPipelineStubs(s1, s2)
    coord = Coordinator(
        config=cfg,
        worker1_stub=backend.worker1_stub,
        worker2_stub=backend.worker2_stub,
        tokenizer=baseline.tokenizer,
        retry_backoff_s=0.0,
    )

    prompt = "In distributed systems, fault tolerance ensures that"
    ref = baseline.generate_greedy(prompt, max_new_tokens=12, stop_on_eos=False, capture_logits=True)
    out = asyncio.run(
        coord.generate(
            prompt,
            max_new_tokens=12,
            stop_on_eos=False,
            capture_logits=(selection_mode == "coordinator"),
            selection_mode=selection_mode,
        )
    )

    assert out["status"] == "COMPLETED"
    assert out["generated_ids"] == ref["generated_ids"]
    assert out["total_activation_bytes"] > 0
    if selection_mode == "coordinator":
        assert out["total_logits_bytes"] > 0
        assert len(out["logits_history"]) == 12
        for got, exp in zip(out["logits_history"], ref["logits_history"]):
            assert torch.allclose(got, exp, atol=1e-5)
    else:
        assert out["total_logits_bytes"] == 0
    # Caches cleaned up after completion
    assert s1.active_entries() == 0 and s2.active_entries() == 0


def test_coordinator_concurrent_batch_and_csv_export(cfg, baseline, partitions, tmp_path):
    s1, s2 = partitions
    s1.clear()
    s2.clear()
    backend = InProcessPipelineStubs(s1, s2)
    coord = Coordinator(
        config=cfg,
        worker1_stub=backend.worker1_stub,
        worker2_stub=backend.worker2_stub,
        tokenizer=baseline.tokenizer,
        retry_backoff_s=0.0,
    )

    prompts = [
        "Pipeline parallelism divides transformer layers across",
        "Bounded queues and backpressure prevent overload by",
        "Key-value caching avoids recomputing attention over",
        "Epoch fencing rejects stale worker responses when",
    ]
    refs = [baseline.generate_greedy(p, max_new_tokens=8, stop_on_eos=False) for p in prompts]
    outs = asyncio.run(
        coord.generate_batch(prompts, max_new_tokens=8, stop_on_eos=False, selection_mode="worker")
    )

    for out, ref in zip(outs, refs):
        assert out["status"] == "COMPLETED"
        assert out["generated_ids"] == ref["generated_ids"]

    summary = coord.metrics.summary()
    assert summary["completed_requests"] == 4
    assert summary["total_generated_tokens"] == 32
    req_csv = coord.metrics.export_requests_csv(tmp_path / "requests.csv")
    step_csv = coord.metrics.export_steps_csv(tmp_path / "steps.csv")
    assert req_csv.exists() and step_csv.exists()
    assert len(req_csv.read_text(encoding="utf-8").splitlines()) == 5  # header + 4 rows


def test_coordinator_worker2_crash_replay_and_exact_recovery(cfg, baseline, partitions):
    s1, s2 = partitions
    s1.clear()
    s2.clear()
    backend = InProcessPipelineStubs(s1, s2)
    # Inject a Worker 2 crash + epoch bump on decode step 4
    backend.fail_on_calls[("req-crash-test", 4)] = "w2_crash"

    coord = Coordinator(
        config=cfg,
        worker1_stub=backend.worker1_stub,
        worker2_stub=backend.worker2_stub,
        tokenizer=baseline.tokenizer,
        retry_backoff_s=0.0,
    )

    prompt = "Distributed inference recovery replays committed tokens so"
    ref = baseline.generate_greedy(prompt, max_new_tokens=10, stop_on_eos=False)
    out = asyncio.run(
        coord.generate(
            prompt,
            max_new_tokens=10,
            stop_on_eos=False,
            selection_mode="worker",
            request_id="req-crash-test",
        )
    )

    assert out["status"] == "COMPLETED"
    assert out["retry_count"] == 1
    assert out["recovery_time_s"] >= 0.0
    assert coord.worker2_epoch == 2
    assert out["generated_ids"] == ref["generated_ids"]
    # Both workers had their caches reset during recovery
    assert ("w1", "req-crash-test", "att-1") in backend.reset_calls
    assert ("w2", "req-crash-test", "att-1") in backend.reset_calls


def test_coordinator_retry_budget_exhausted_fails_explicitly(cfg, baseline, partitions):
    s1, s2 = partitions
    s1.clear()
    s2.clear()
    backend = InProcessPipelineStubs(s1, s2)
    backend.fail_on_calls[("req-exhaust", 0)] = "always_overloaded"

    coord = Coordinator(
        config=cfg,
        worker1_stub=backend.worker1_stub,
        worker2_stub=backend.worker2_stub,
        tokenizer=baseline.tokenizer,
        retry_backoff_s=0.0,
    )

    out = asyncio.run(
        coord.generate(
            "Overloaded pipeline request",
            max_new_tokens=5,
            selection_mode="worker",
            request_id="req-exhaust",
        )
    )
    assert out["status"] == "FAILED"
    assert out["retry_count"] == cfg["reliability"]["max_retries"]
    assert "Retry budget exhausted" in out["error_message"]


# ---------------------------------------------------------------------------
# Fast offline unit tests (no Hugging Face model download required)
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    eos_token_id = 99

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = [10 + (ord(ch) % 20) for ch in text[:4]]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(t) for t in token_ids)


class _FakePartition:
    """Deterministic stateful cache simulator for offline coordinator testing."""

    def __init__(self, emit_eos_at_pos: int | None = None):
        self.caches: dict[tuple[str, str], list[int]] = {}
        self.emit_eos_at_pos = emit_eos_at_pos

    def active_entries(self) -> int:
        return len(self.caches)

    def clear(self) -> None:
        self.caches.clear()

    def reset(self, request_id: str, attempt_id: str | None = None) -> int:
        if attempt_id:
            return 1 if self.caches.pop((request_id, attempt_id), None) is not None else 0
        keys = [k for k in self.caches if k[0] == request_id]
        for k in keys:
            self.caches.pop(k, None)
        return len(keys)

    def forward(
        self,
        request_id: str,
        attempt_id: str,
        inp: torch.Tensor,
        expected_cache_pos: int | None = None,
        selection_mode: str = "coordinator",
    ):
        key = (request_id, attempt_id)
        hist = self.caches.setdefault(key, [])
        if expected_cache_pos is not None and len(hist) != expected_cache_pos:
            raise ValueError(f"Cache mismatch: expected {expected_cache_pos}, got {len(hist)}")

        if inp.ndim == 2:
            tokens = inp[0].tolist()
            hist.extend(tokens)
            return torch.tensor([[[float(t), 1.0, 2.0, 3.0] for t in tokens]], dtype=torch.float32)

        seq_len = inp.shape[1]
        last_val = int(round(float(inp[0, -1, 0].item())))
        hist.extend([last_val] * seq_len)
        next_tok = (
            99
            if self.emit_eos_at_pos is not None and len(hist) >= self.emit_eos_at_pos
            else ((last_val + len(hist)) % 50)
        )
        if selection_mode == "worker":
            return next_tok
        logits = torch.full((1, 100), -10.0, dtype=torch.float32)
        logits[0, next_tok] = 10.0
        return logits


@pytest.mark.parametrize("selection_mode", ["coordinator", "worker"])
def test_offline_coordinator_modes_eos_and_zero_tokens(cfg, selection_mode):
    s1 = _FakePartition()
    s2 = _FakePartition(emit_eos_at_pos=6)
    backend = InProcessPipelineStubs(s1, s2)
    coord = Coordinator(
        config=cfg,
        worker1_stub=backend.worker1_stub,
        worker2_stub=backend.worker2_stub,
        tokenizer=_FakeTokenizer(),
        retry_backoff_s=0.0,
    )

    zero_out = asyncio.run(coord.generate("abcd", max_new_tokens=0, selection_mode=selection_mode))
    assert zero_out["status"] == "COMPLETED"
    assert zero_out["new_token_ids"] == []

    eos_out = asyncio.run(
        coord.generate("abcd", max_new_tokens=10, stop_on_eos=True, selection_mode=selection_mode)
    )
    assert eos_out["status"] == "COMPLETED"
    assert eos_out["new_token_ids"][-1] == 99
    assert len(eos_out["new_token_ids"]) == 3
    assert s1.active_entries() == 0 and s2.active_entries() == 0


def test_offline_coordinator_crash_replay_and_exhaustion(cfg, tmp_path):
    ref_backend = InProcessPipelineStubs(_FakePartition(), _FakePartition())
    ref_coord = Coordinator(
        config=cfg,
        worker1_stub=ref_backend.worker1_stub,
        worker2_stub=ref_backend.worker2_stub,
        tokenizer=_FakeTokenizer(),
        retry_backoff_s=0.0,
    )
    ref_out = asyncio.run(
        ref_coord.generate("abcd", max_new_tokens=8, stop_on_eos=False, selection_mode="worker")
    )

    crash_backend = InProcessPipelineStubs(_FakePartition(), _FakePartition())
    crash_backend.fail_on_calls[("req-offline-crash", 4)] = "w2_crash"
    crash_coord = Coordinator(
        config=cfg,
        worker1_stub=crash_backend.worker1_stub,
        worker2_stub=crash_backend.worker2_stub,
        tokenizer=_FakeTokenizer(),
        retry_backoff_s=0.0,
    )
    out = asyncio.run(
        crash_coord.generate(
            "abcd",
            max_new_tokens=8,
            stop_on_eos=False,
            selection_mode="worker",
            request_id="req-offline-crash",
        )
    )
    assert out["status"] == "COMPLETED"
    assert out["retry_count"] == 1
    assert crash_coord.worker2_epoch == 2
    assert out["generated_ids"] == ref_out["generated_ids"]
    assert ("w1", "req-offline-crash", "att-1") in crash_backend.reset_calls
    assert ("w2", "req-offline-crash", "att-1") in crash_backend.reset_calls

    crash_backend.fail_on_calls[("req-offline-exhaust", 0)] = "always_overloaded"
    failed = asyncio.run(
        crash_coord.generate(
            "abcd",
            max_new_tokens=5,
            selection_mode="worker",
            request_id="req-offline-exhaust",
        )
    )
    assert failed["status"] == "FAILED"
    assert failed["retry_count"] == cfg["reliability"]["max_retries"]

    req_csv = crash_coord.metrics.export_requests_csv(tmp_path / "requests.csv")
    step_csv = crash_coord.metrics.export_steps_csv(tmp_path / "steps.csv")
    assert req_csv.exists() and step_csv.exists()
