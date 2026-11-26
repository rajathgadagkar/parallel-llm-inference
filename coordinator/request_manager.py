"""Per-request state machine, admission control, and ordered token commitment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

import torch


class RequestState(str, Enum):
    CREATED = "CREATED"
    PREFILLING = "PREFILLING"
    ACTIVE = "ACTIVE"
    REPLAYING = "REPLAYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class OverloadError(RuntimeError):
    """Raised when coordinator admission control rejects a request due to capacity."""


class StaleResponseError(RuntimeError):
    """Raised when a response does not match the request's current attempt or step."""


@dataclass
class RequestRecord:
    request_id: str
    prompt: str
    prompt_ids: list[int]
    max_new_tokens: int
    stop_on_eos: bool = True
    selection_mode: str = "coordinator"
    capture_logits: bool = False
    attempt_num: int = 1
    next_step_id: int = 0
    committed_tokens: list[int] = field(default_factory=list)
    logits_history: list[torch.Tensor] = field(default_factory=list)
    state: RequestState = RequestState.CREATED
    retry_count: int = 0
    error_message: str = ""
    created_at_s: float = field(default_factory=time.perf_counter)
    completed_at_s: float | None = None

    @property
    def attempt_id(self) -> str:
        return f"att-{self.attempt_num}"

    @property
    def is_terminal(self) -> bool:
        return self.state in (RequestState.COMPLETED, RequestState.FAILED)

    @property
    def cache_position(self) -> int:
        """Expected KV-cache length on both workers before the next non-replay step."""
        if not self.committed_tokens:
            return 0
        return len(self.prompt_ids) + len(self.committed_tokens) - 1

    @property
    def generated_ids(self) -> list[int]:
        return self.prompt_ids + self.committed_tokens


class RequestManager:
    """Tracks active/completed inference requests with admission control and ordering."""

    def __init__(self, max_concurrent_requests: int = 8):
        if max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be >= 1, got {max_concurrent_requests}"
            )
        self.max_concurrent_requests = max_concurrent_requests
        self._records: dict[str, RequestRecord] = {}

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._records.values() if not r.is_terminal)

    def get(self, request_id: str) -> RequestRecord:
        if request_id not in self._records:
            raise KeyError(f"Unknown request_id: {request_id}")
        return self._records[request_id]

    def create_request(
        self,
        prompt: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        stop_on_eos: bool = True,
        selection_mode: str = "coordinator",
        capture_logits: bool = False,
        request_id: str | None = None,
    ) -> RequestRecord:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if not prompt_ids:
            raise ValueError("prompt_ids cannot be empty")
        if selection_mode not in ("coordinator", "worker"):
            raise ValueError(
                f"selection_mode must be 'coordinator' or 'worker', got '{selection_mode}'"
            )
        if self.active_count >= self.max_concurrent_requests:
            raise OverloadError(
                f"Coordinator at capacity ({self.max_concurrent_requests} active requests)"
            )

        rid = request_id or f"req-{uuid.uuid4().hex[:12]}"
        if rid in self._records:
            raise ValueError(f"Duplicate request_id: {rid}")

        record = RequestRecord(
            request_id=rid,
            prompt=prompt,
            prompt_ids=list(prompt_ids),
            max_new_tokens=max_new_tokens,
            stop_on_eos=stop_on_eos,
            selection_mode=selection_mode,
            capture_logits=capture_logits,
        )
        if max_new_tokens == 0:
            record.state = RequestState.COMPLETED
            record.completed_at_s = record.created_at_s
        else:
            record.state = RequestState.PREFILLING
        self._records[rid] = record
        return record

    def commit_step(
        self,
        request_id: str,
        attempt_id: str,
        step_id: int,
        token_id: int,
        eos_token_id: int | None = None,
        logits: torch.Tensor | None = None,
    ) -> RequestRecord:
        """Commit a single generated token exactly once in strict step order."""
        record = self.get(request_id)
        if record.is_terminal:
            raise StaleResponseError(
                f"Cannot commit step {step_id} for terminal request {request_id} ({record.state})"
            )
        if attempt_id != record.attempt_id:
            raise StaleResponseError(
                f"Stale attempt_id '{attempt_id}' for {request_id} "
                f"(expected '{record.attempt_id}')"
            )
        if step_id != record.next_step_id:
            raise StaleResponseError(
                f"Out-of-order or duplicate step_id {step_id} for {request_id} "
                f"(expected {record.next_step_id})"
            )

        record.committed_tokens.append(int(token_id))
        if record.capture_logits and logits is not None:
            record.logits_history.append(logits.detach().cpu())
        record.next_step_id += 1

        reached_eos = (
            record.stop_on_eos and eos_token_id is not None and int(token_id) == eos_token_id
        )
        reached_limit = len(record.committed_tokens) >= record.max_new_tokens

        if reached_eos or reached_limit:
            record.state = RequestState.COMPLETED
            record.completed_at_s = time.perf_counter()
        else:
            record.state = RequestState.ACTIVE
        return record

    def begin_replay(self, request_id: str, max_retries: int) -> RequestRecord:
        """Advance attempt_id and transition to REPLAYING, or fail if retry budget is spent."""
        record = self.get(request_id)
        if record.is_terminal:
            raise RuntimeError(f"Cannot replay terminal request {request_id} ({record.state})")
        if record.retry_count >= max_retries:
            self.mark_failed(
                request_id,
                f"Retry budget exhausted after {record.retry_count} retries",
            )
            return record

        record.retry_count += 1
        record.attempt_num += 1
        record.state = RequestState.REPLAYING
        return record

    def finish_replay(self, request_id: str) -> RequestRecord:
        """Mark cache reconstruction complete and return to PREFILLING or ACTIVE."""
        record = self.get(request_id)
        if record.state != RequestState.REPLAYING:
            raise RuntimeError(f"Request {request_id} is not REPLAYING (state={record.state})")
        record.state = RequestState.ACTIVE if record.committed_tokens else RequestState.PREFILLING
        return record

    def mark_failed(self, request_id: str, error_message: str) -> RequestRecord:
        record = self.get(request_id)
        record.state = RequestState.FAILED
        record.error_message = error_message
        record.completed_at_s = time.perf_counter()
        return record
