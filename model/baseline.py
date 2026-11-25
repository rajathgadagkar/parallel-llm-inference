import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_STR_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype not in _STR_TO_TORCH_DTYPE:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Supported: {list(_STR_TO_TORCH_DTYPE.keys())}"
        )
    return _STR_TO_TORCH_DTYPE[dtype]


class BaselineInference:
    """Single-process deterministic greedy decoding baseline."""

    def __init__(
        self,
        model_name: str = "distilbert/distilgpt2",
        device: str = "cpu",
        dtype: str | torch.dtype = "float32",
    ):
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(
            device=self.device, dtype=self.dtype
        )
        self.model.eval()

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        stop_on_eos: bool = True,
        capture_logits: bool = False,
    ) -> dict:
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        if not prompt:
            raise ValueError("Prompt cannot be empty")

        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        if input_ids.shape[1] == 0:
            raise ValueError("Prompt produced 0 tokens after tokenization")

        prompt_ids = input_ids[0].tolist()
        if max_new_tokens == 0:
            return {
                "prompt_ids": prompt_ids,
                "new_token_ids": [],
                "generated_ids": prompt_ids,
                "generated_text": self.tokenizer.decode(prompt_ids),
                "logits_history": [],
                "prefill_time_s": 0.0,
                "decode_token_latencies_s": [],
                "ttft_s": 0.0,
                "total_time_s": 0.0,
            }

        eos_id = self.tokenizer.eos_token_id
        new_token_ids: list[int] = []
        logits_history: list[torch.Tensor] = []
        decode_latencies: list[float] = []

        self._sync_device()
        start_t = time.perf_counter()

        with torch.no_grad():
            # Step 0 (Prefill)
            t0 = time.perf_counter()
            out = self.model(input_ids=input_ids, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            if capture_logits:
                logits_history.append(logits.detach().cpu())
            next_tok = logits.argmax(dim=-1, keepdim=True)
            next_id = int(next_tok.item())
            new_token_ids.append(next_id)
            self._sync_device()
            prefill_time_s = time.perf_counter() - t0
            ttft_s = time.perf_counter() - start_t

            # Steps 1..max_new_tokens-1 (Decode)
            if not (stop_on_eos and eos_id is not None and next_id == eos_id):
                for _ in range(max_new_tokens - 1):
                    t0 = time.perf_counter()
                    out = self.model(input_ids=next_tok, past_key_values=pkv, use_cache=True)
                    pkv = out.past_key_values
                    logits = out.logits[:, -1, :]
                    if capture_logits:
                        logits_history.append(logits.detach().cpu())
                    next_tok = logits.argmax(dim=-1, keepdim=True)
                    next_id = int(next_tok.item())
                    new_token_ids.append(next_id)
                    self._sync_device()
                    decode_latencies.append(time.perf_counter() - t0)

                    if stop_on_eos and eos_id is not None and next_id == eos_id:
                        break

        total_time_s = time.perf_counter() - start_t
        generated_ids = prompt_ids + new_token_ids

        return {
            "prompt_ids": prompt_ids,
            "new_token_ids": new_token_ids,
            "generated_ids": generated_ids,
            "generated_text": self.tokenizer.decode(generated_ids),
            "logits_history": logits_history,
            "prefill_time_s": prefill_time_s,
            "decode_token_latencies_s": decode_latencies,
            "ttft_s": ttft_s,
            "total_time_s": total_time_s,
        }