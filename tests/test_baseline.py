import pytest
import torch
from common.config import load_config
from model.baseline import BaselineInference


@pytest.fixture(scope="module")
def baseline():
    cfg = load_config("config.yaml")
    return BaselineInference(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
    )


def test_greedy_determinism_matches_hf_generate(baseline):
    prompt = "In distributed systems, fault tolerance ensures that"
    out = baseline.generate_greedy(prompt, max_new_tokens=12, capture_logits=True)
    out_repeat = baseline.generate_greedy(prompt, max_new_tokens=12, capture_logits=False)

    assert out["generated_ids"] == out_repeat["generated_ids"]
    assert len(out["new_token_ids"]) == 12
    assert out["generated_ids"] == out["prompt_ids"] + out["new_token_ids"]
    assert len(out["logits_history"]) == 12
    assert all(l.device.type == "cpu" for l in out["logits_history"])
    assert out_repeat["logits_history"] == []
    assert out["prefill_time_s"] > 0
    assert len(out["decode_token_latencies_s"]) == 11

    inp = baseline.tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        ref = baseline.model.generate(
            inp,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=baseline.tokenizer.eos_token_id,
        )
    assert out["generated_ids"] == ref[0].tolist()


def test_max_new_tokens_zero(baseline):
    prompt = "Hello world"
    out = baseline.generate_greedy(prompt, max_new_tokens=0)
    assert out["new_token_ids"] == []
    assert out["generated_ids"] == out["prompt_ids"]
    assert out["prefill_time_s"] == 0.0
    assert out["decode_token_latencies_s"] == []


def test_invalid_inputs_raise(baseline):
    with pytest.raises(ValueError, match="max_new_tokens"):
        baseline.generate_greedy("Hello", max_new_tokens=-1)
    with pytest.raises(ValueError, match="empty"):
        baseline.generate_greedy("", max_new_tokens=5)


def test_stop_on_eos(baseline):
    prompt = "Distributed systems"
    first_out = baseline.generate_greedy(prompt, max_new_tokens=5, stop_on_eos=False)
    first_tok = first_out["new_token_ids"][0]

    orig_tok = baseline.tokenizer

    class _EOSProxy:
        def __init__(self, wrapped, eos_id):
            self._wrapped = wrapped
            self.eos_token_id = eos_id

        def __call__(self, *args, **kwargs):
            return self._wrapped(*args, **kwargs)

        def decode(self, *args, **kwargs):
            return self._wrapped.decode(*args, **kwargs)

    baseline.tokenizer = _EOSProxy(orig_tok, first_tok)
    try:
        stopped = baseline.generate_greedy(prompt, max_new_tokens=5, stop_on_eos=True)
        continued = baseline.generate_greedy(prompt, max_new_tokens=5, stop_on_eos=False)
    finally:
        baseline.tokenizer = orig_tok

    assert stopped["new_token_ids"] == [first_tok]
    assert len(continued["new_token_ids"]) == 5