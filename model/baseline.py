import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaselineInference:
    """Single-process deterministic greedy decoding baseline."""

    def __init__(self, model_name: str = "distilbert/distilgpt2", device: str = "cpu"):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate_greedy(self, prompt: str, max_new_tokens: int = 32) -> dict:
        start_t = time.perf_counter()
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        generated = input_ids.clone()
        logits_history = []
        token_latencies = []

        with torch.no_grad():
            # Step 0 (Prefill): process all prompt tokens at once
            t0 = time.perf_counter()
            out = self.model(input_ids=input_ids, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :]
            logits_history.append(logits)
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=-1)
            token_latencies.append(time.perf_counter() - t0)
            ttft_s = time.perf_counter() - start_t

            # Steps 1..N-1 (Decode): feed 1 token at a time using KV cache
            for _ in range(max_new_tokens - 1):
                t0 = time.perf_counter()
                out = self.model(input_ids=next_tok, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                logits = out.logits[:, -1, :]
                logits_history.append(logits)
                next_tok = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=-1)
                token_latencies.append(time.perf_counter() - t0)

        return {
            "generated_ids": generated[0].tolist(),
            "generated_text": self.tokenizer.decode(generated[0]),
            "logits_history": logits_history,
            "ttft_s": ttft_s,
            "total_time_s": time.perf_counter() - start_t,
            "token_latencies_s": token_latencies,
        }