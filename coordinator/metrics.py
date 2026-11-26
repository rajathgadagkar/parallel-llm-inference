"""Metrics collection, aggregation, and CSV export for pipeline evaluation."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class StepMetrics:
    request_id: str
    attempt_id: str
    step_id: int
    is_prefill: bool
    is_replay: bool
    step_latency_ms: float
    stage1_compute_ms: float = 0.0
    stage1_queue_wait_ms: float = 0.0
    stage2_compute_ms: float = 0.0
    stage2_queue_wait_ms: float = 0.0
    activation_payload_bytes: int = 0
    rpc_control_bytes: int = 0
    logits_payload_bytes: int = 0


@dataclass
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    ttft_s: float
    prefill_time_s: float
    decode_token_latencies_s: list[float] = field(default_factory=list)
    total_time_s: float = 0.0
    total_activation_bytes: int = 0
    total_rpc_control_bytes: int = 0
    total_logits_bytes: int = 0
    retry_count: int = 0
    recovery_time_s: float = 0.0
    status: str = "COMPLETED"
    step_metrics: list[StepMetrics] = field(default_factory=list)


class MetricsCollector:
    """Aggregates request and step-level measurements and writes raw CSVs."""

    def __init__(self) -> None:
        self.requests: list[RequestMetrics] = []

    def record_request(self, metrics: RequestMetrics) -> None:
        self.requests.append(metrics)

    def clear(self) -> None:
        self.requests.clear()

    def summary(self, wall_time_s: float | None = None) -> dict:
        completed = [r for r in self.requests if r.status == "COMPLETED"]
        failed = [r for r in self.requests if r.status != "COMPLETED"]

        all_decode_ms = [
            lat * 1000.0 for r in completed for lat in r.decode_token_latencies_s
        ]
        all_steps = [s for r in self.requests for s in r.step_metrics]

        total_gen_tokens = sum(r.generated_tokens for r in completed)
        effective_wall_s = (
            wall_time_s
            if wall_time_s is not None and wall_time_s > 0
            else sum(r.total_time_s for r in completed)
        )

        return {
            "total_requests": len(self.requests),
            "completed_requests": len(completed),
            "failed_requests": len(failed),
            "total_generated_tokens": total_gen_tokens,
            "mean_ttft_ms": (
                float(np.mean([r.ttft_s * 1000.0 for r in completed])) if completed else 0.0
            ),
            "p50_tpot_ms": float(np.percentile(all_decode_ms, 50)) if all_decode_ms else 0.0,
            "p95_tpot_ms": float(np.percentile(all_decode_ms, 95)) if all_decode_ms else 0.0,
            "aggregate_tokens_per_sec": (
                total_gen_tokens / effective_wall_s if effective_wall_s > 0 else 0.0
            ),
            "requests_per_sec": (
                len(completed) / effective_wall_s if effective_wall_s > 0 else 0.0
            ),
            "mean_stage1_compute_ms": (
                float(np.mean([s.stage1_compute_ms for s in all_steps])) if all_steps else 0.0
            ),
            "mean_stage2_compute_ms": (
                float(np.mean([s.stage2_compute_ms for s in all_steps])) if all_steps else 0.0
            ),
            "mean_stage1_queue_wait_ms": (
                float(np.mean([s.stage1_queue_wait_ms for s in all_steps])) if all_steps else 0.0
            ),
            "mean_stage2_queue_wait_ms": (
                float(np.mean([s.stage2_queue_wait_ms for s in all_steps])) if all_steps else 0.0
            ),
            "total_activation_bytes": sum(r.total_activation_bytes for r in self.requests),
            "total_rpc_control_bytes": sum(r.total_rpc_control_bytes for r in self.requests),
            "total_logits_bytes": sum(r.total_logits_bytes for r in self.requests),
            "total_retries": sum(r.retry_count for r in self.requests),
            "total_recovery_time_s": sum(r.recovery_time_s for r in self.requests),
        }

    def export_requests_csv(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "request_id",
            "status",
            "prompt_tokens",
            "generated_tokens",
            "ttft_s",
            "prefill_time_s",
            "mean_decode_ms",
            "p95_decode_ms",
            "total_time_s",
            "total_activation_bytes",
            "total_rpc_control_bytes",
            "total_logits_bytes",
            "retry_count",
            "recovery_time_s",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.requests:
                dec_ms = [x * 1000.0 for x in r.decode_token_latencies_s]
                writer.writerow(
                    {
                        "request_id": r.request_id,
                        "status": r.status,
                        "prompt_tokens": r.prompt_tokens,
                        "generated_tokens": r.generated_tokens,
                        "ttft_s": f"{r.ttft_s:.6f}",
                        "prefill_time_s": f"{r.prefill_time_s:.6f}",
                        "mean_decode_ms": f"{float(np.mean(dec_ms)):.4f}" if dec_ms else "0.0000",
                        "p95_decode_ms": (
                            f"{float(np.percentile(dec_ms, 95)):.4f}" if dec_ms else "0.0000"
                        ),
                        "total_time_s": f"{r.total_time_s:.6f}",
                        "total_activation_bytes": r.total_activation_bytes,
                        "total_rpc_control_bytes": r.total_rpc_control_bytes,
                        "total_logits_bytes": r.total_logits_bytes,
                        "retry_count": r.retry_count,
                        "recovery_time_s": f"{r.recovery_time_s:.6f}",
                    }
                )
        return path

    def export_steps_csv(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "request_id",
            "attempt_id",
            "step_id",
            "is_prefill",
            "is_replay",
            "step_latency_ms",
            "stage1_compute_ms",
            "stage1_queue_wait_ms",
            "stage2_compute_ms",
            "stage2_queue_wait_ms",
            "activation_payload_bytes",
            "rpc_control_bytes",
            "logits_payload_bytes",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.requests:
                for s in r.step_metrics:
                    writer.writerow(
                        {
                            "request_id": s.request_id,
                            "attempt_id": s.attempt_id,
                            "step_id": s.step_id,
                            "is_prefill": s.is_prefill,
                            "is_replay": s.is_replay,
                            "step_latency_ms": f"{s.step_latency_ms:.4f}",
                            "stage1_compute_ms": f"{s.stage1_compute_ms:.4f}",
                            "stage1_queue_wait_ms": f"{s.stage1_queue_wait_ms:.4f}",
                            "stage2_compute_ms": f"{s.stage2_compute_ms:.4f}",
                            "stage2_queue_wait_ms": f"{s.stage2_queue_wait_ms:.4f}",
                            "activation_payload_bytes": s.activation_payload_bytes,
                            "rpc_control_bytes": s.rpc_control_bytes,
                            "logits_payload_bytes": s.logits_payload_bytes,
                        }
                    )
        return path
