import threading
import time
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from model.baseline import _resolve_dtype


class _BasePartition:
    def __init__(self, ttl_seconds: float = 300.0):
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self.ttl_seconds = ttl_seconds
        self._kv_store: dict[tuple[str, str], DynamicCache] = {}
        self._last_used: dict[tuple[str, str], float] = {}
        self._key_locks: dict[tuple[str, str], threading.Lock] = {}
        self._store_lock = threading.Lock()

    def _validate_ids(self, request_id: str, attempt_id: str) -> tuple[str, str]:
        if not request_id or not attempt_id:
            raise ValueError("request_id and attempt_id must be non-empty strings")
        return (request_id, attempt_id)

    def _get_lock(self, cache_key: tuple[str, str]) -> threading.Lock:
        with self._store_lock:
            if cache_key not in self._key_locks:
                self._key_locks[cache_key] = threading.Lock()
            return self._key_locks[cache_key]

    def _prune_lock_unlocked(self, key: tuple[str, str]) -> None:
        lock = self._key_locks.get(key)
        if lock is not None and not lock.locked():
            self._key_locks.pop(key, None)

    def reset(self, request_id: str, attempt_id: str | None = None) -> int:
        """Reset KV cache for (request_id, attempt_id) or all attempts of request_id."""
        with self._store_lock:
            if attempt_id is not None and attempt_id != "":
                key = (request_id, attempt_id)
                removed = 1 if key in self._kv_store else 0
                self._kv_store.pop(key, None)
                self._last_used.pop(key, None)
                self._prune_lock_unlocked(key)
                return removed
            keys = [k for k in self._kv_store if k[0] == request_id]
            for k in keys:
                self._kv_store.pop(k, None)
                self._last_used.pop(k, None)
                self._prune_lock_unlocked(k)
            return len(keys)

    def clear(self) -> None:
        with self._store_lock:
            keys = list(self._kv_store.keys())
            self._kv_store.clear()
            self._last_used.clear()
            for k in keys:
                self._prune_lock_unlocked(k)

    def evict_expired(self, now: float | None = None) -> int:
        current = time.monotonic() if now is None else now
        with self._store_lock:
            expired = [k for k, ts in self._last_used.items() if (current - ts) > self.ttl_seconds]
            for k in expired:
                self._kv_store.pop(k, None)
                self._last_used.pop(k, None)
                self._prune_lock_unlocked(k)
            return len(expired)

    def active_entries(self) -> int:
        with self._store_lock:
            return len(self._kv_store)


class Stage1Partition(_BasePartition):
    """Worker 1 model partition: Embeddings + Transformer layers [0 .. split_layer - 1]."""

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        split_layer: int = 3,
        device: str = "cpu",
        dtype: str | torch.dtype = "float32",
        ttl_seconds: float = 300.0,
    ):
        super().__init__(ttl_seconds=ttl_seconds)
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype)

        full_model = AutoModelForCausalLM.from_pretrained(model_name).to(
            device=self.device, dtype=self.dtype
        )
        full_model.eval()
        num_layers = len(full_model.transformer.h)
        if not (1 <= split_layer < num_layers):
            raise ValueError(f"split_layer must be in [1, {num_layers - 1}], got {split_layer}")

        self.config = full_model.config
        self.split_layer = split_layer
        self.wte = full_model.transformer.wte
        self.wpe = full_model.transformer.wpe
        self.drop = full_model.transformer.drop
        self.blocks = full_model.transformer.h[:split_layer]
        del full_model

    def get_cache_length(self, request_id: str, attempt_id: str) -> int:
        key = self._validate_ids(request_id, attempt_id)
        with self._store_lock:
            cache = self._kv_store.get(key)
            return cache.get_seq_length(layer_idx=0) if cache is not None else 0

    def prefill(self, request_id: str, attempt_id: str, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.forward(request_id, attempt_id, input_ids, expected_cache_pos=0)

    def decode(
        self,
        request_id: str,
        attempt_id: str,
        token_id: int | torch.LongTensor,
        expected_cache_pos: int | None = None,
    ) -> torch.Tensor:
        inp = torch.tensor([[token_id]], dtype=torch.long) if isinstance(token_id, int) else token_id
        return self.forward(request_id, attempt_id, inp, expected_cache_pos=expected_cache_pos)

    def forward(
        self,
        request_id: str,
        attempt_id: str,
        input_ids: torch.LongTensor,
        expected_cache_pos: int | None = None,
    ) -> torch.Tensor:
        key = self._validate_ids(request_id, attempt_id)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError(f"input_ids must have shape [1, seq_len >= 1], got {tuple(input_ids.shape)}")

        lock = self._get_lock(key)
        with lock:
            with self._store_lock:
                if key not in self._kv_store:
                    self._kv_store[key] = DynamicCache(config=self.config)
                cache = self._kv_store[key]
                self._last_used[key] = time.monotonic()

            input_ids = input_ids.to(self.device)
            past_seen = cache.get_seq_length(layer_idx=0)
            if expected_cache_pos is not None and past_seen != expected_cache_pos:
                raise ValueError(
                    f"[Stage1] Cache mismatch for {key}: expected {expected_cache_pos}, got {past_seen}"
                )

            seq_len = input_ids.shape[1]
            position_ids = (torch.arange(seq_len, device=self.device) + past_seen).unsqueeze(0)

            with torch.no_grad():
                inputs_embeds = self.wte(input_ids)
                position_embeds = self.wpe(position_ids)
                hidden_states = self.drop(inputs_embeds + position_embeds)

                causal_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,
                    past_key_values=cache,
                    position_ids=position_ids,
                    layer_idx=0,
                )
                for block in self.blocks:
                    hidden_states = block(
                        hidden_states,
                        past_key_values=cache,
                        attention_mask=causal_mask,
                        use_cache=True,
                        position_ids=position_ids,
                    )
            return hidden_states


class Stage2Partition(_BasePartition):
    """Worker 2 model partition: Transformer layers [split_layer .. end] + ln_f + lm_head."""

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        split_layer: int = 3,
        device: str = "cpu",
        dtype: str | torch.dtype = "float32",
        ttl_seconds: float = 300.0,
    ):
        super().__init__(ttl_seconds=ttl_seconds)
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype)

        full_model = AutoModelForCausalLM.from_pretrained(model_name).to(
            device=self.device, dtype=self.dtype
        )
        full_model.eval()
        num_layers = len(full_model.transformer.h)
        if not (1 <= split_layer < num_layers):
            raise ValueError(f"split_layer must be in [1, {num_layers - 1}], got {split_layer}")

        self.config = full_model.config
        self.split_layer = split_layer
        self.blocks = full_model.transformer.h[split_layer:]
        self.ln_f = full_model.transformer.ln_f
        self.lm_head = full_model.lm_head
        del full_model

    def get_cache_length(self, request_id: str, attempt_id: str) -> int:
        key = self._validate_ids(request_id, attempt_id)
        with self._store_lock:
            cache = self._kv_store.get(key)
            return cache.get_seq_length(layer_idx=self.split_layer) if cache is not None else 0

    def prefill(
        self,
        request_id: str,
        attempt_id: str,
        hidden_states: torch.Tensor,
        selection_mode: str = "coordinator",
    ) -> torch.Tensor | int:
        return self.forward(
            request_id,
            attempt_id,
            hidden_states,
            expected_cache_pos=0,
            selection_mode=selection_mode,
        )

    def decode(
        self,
        request_id: str,
        attempt_id: str,
        hidden_states: torch.Tensor,
        expected_cache_pos: int | None = None,
        selection_mode: str = "coordinator",
    ) -> torch.Tensor | int:
        return self.forward(
            request_id,
            attempt_id,
            hidden_states,
            expected_cache_pos=expected_cache_pos,
            selection_mode=selection_mode,
        )

    def forward(
        self,
        request_id: str,
        attempt_id: str,
        hidden_states: torch.Tensor,
        expected_cache_pos: int | None = None,
        selection_mode: str = "coordinator",
    ) -> torch.Tensor | int:
        if selection_mode not in ("coordinator", "worker"):
            raise ValueError(f"selection_mode must be 'coordinator' or 'worker', got '{selection_mode}'")
        key = self._validate_ids(request_id, attempt_id)
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or hidden_states.shape[1] == 0:
            raise ValueError(
                f"hidden_states must have shape [1, seq_len >= 1, hidden_dim], got {tuple(hidden_states.shape)}"
            )

        lock = self._get_lock(key)
        with lock:
            with self._store_lock:
                if key not in self._kv_store:
                    self._kv_store[key] = DynamicCache(config=self.config)
                cache = self._kv_store[key]
                self._last_used[key] = time.monotonic()

            hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
            past_seen = cache.get_seq_length(layer_idx=self.split_layer)
            if expected_cache_pos is not None and past_seen != expected_cache_pos:
                raise ValueError(
                    f"[Stage2] Cache mismatch for {key}: expected {expected_cache_pos}, got {past_seen}"
                )

            seq_len = hidden_states.shape[1]
            position_ids = (torch.arange(seq_len, device=self.device) + past_seen).unsqueeze(0)

            with torch.no_grad():
                causal_mask = create_causal_mask(
                    config=self.config,
                    inputs_embeds=hidden_states,
                    attention_mask=None,
                    past_key_values=cache,
                    position_ids=position_ids,
                    layer_idx=self.split_layer,
                )
                for block in self.blocks:
                    hidden_states = block(
                        hidden_states,
                        past_key_values=cache,
                        attention_mask=causal_mask,
                        use_cache=True,
                        position_ids=position_ids,
                    )
                hidden_states = self.ln_f(hidden_states)
                logits = self.lm_head(hidden_states[:, -1, :])

            if selection_mode == "worker":
                return int(logits.argmax(dim=-1).item())
            return logits