import time
import pytest
import torch

from common.config import load_config
from common.serialization import bytes_to_tensor, tensor_to_bytes
from model.baseline import BaselineInference
from model.partition import Stage1Partition, Stage2Partition
from proto.generated import inference_pb2


@pytest.fixture(scope="module")
def cfg():
    return load_config("config.yaml")


@pytest.fixture(scope="module")
def baseline(cfg):
    return BaselineInference(
        model_name=cfg["model"]["name"],
        dtype=cfg["model"]["dtype"],
    )


@pytest.mark.parametrize("split_layer", [2, 3, 4])
def test_split_layers_match_baseline_logits_and_tokens(cfg, baseline, split_layer):
    prompt = "In distributed systems, fault tolerance ensures that"
    max_new_tokens = 16
    ref = baseline.generate_greedy(
        prompt, max_new_tokens=max_new_tokens, stop_on_eos=False, capture_logits=True
    )

    s1 = Stage1Partition(cfg["model"]["name"], split_layer=split_layer, dtype=cfg["model"]["dtype"])
    s2 = Stage2Partition(cfg["model"]["name"], split_layer=split_layer, dtype=cfg["model"]["dtype"])

    inp = baseline.tokenizer(prompt, return_tensors="pt")["input_ids"]
    gen = inp[0].tolist()

    h = s1.prefill("r1", "att1", inp)
    logits = s2.prefill("r1", "att1", h, selection_mode="coordinator")
    assert torch.allclose(logits, ref["logits_history"][0], atol=1e-5)
    tok = int(logits.argmax(dim=-1).item())
    gen.append(tok)

    for step in range(1, max_new_tokens):
        pos = inp.shape[1] + step - 1
        h = s1.decode("r1", "att1", tok, expected_cache_pos=pos)
        logits = s2.decode("r1", "att1", h, expected_cache_pos=pos, selection_mode="coordinator")
        assert torch.allclose(logits, ref["logits_history"][step], atol=1e-5)
        tok = int(logits.argmax(dim=-1).item())
        gen.append(tok)

    assert gen == ref["generated_ids"]


def test_mode_b_worker_token_selection(cfg, baseline):
    prompt = "The quick brown fox jumps over"
    max_new_tokens = 16
    ref = baseline.generate_greedy(prompt, max_new_tokens=max_new_tokens, stop_on_eos=False)

    s1 = Stage1Partition(cfg["model"]["name"], split_layer=cfg["model"]["split_layer"])
    s2 = Stage2Partition(cfg["model"]["name"], split_layer=cfg["model"]["split_layer"])

    inp = baseline.tokenizer(prompt, return_tensors="pt")["input_ids"]
    gen = inp[0].tolist()

    h = s1.prefill("mode-b", "att1", inp)
    tok = s2.prefill("mode-b", "att1", h, selection_mode="worker")
    assert isinstance(tok, int)
    gen.append(tok)

    for _ in range(1, max_new_tokens):
        h = s1.decode("mode-b", "att1", tok)
        tok = s2.decode("mode-b", "att1", h, selection_mode="worker")
        gen.append(tok)

    assert gen == ref["generated_ids"]


def test_concurrent_attempt_isolation_and_replay_reset(cfg, baseline):
    prompt_a = "Pipeline parallelism divides transformer layers across"
    prompt_b = "Bounded queues and backpressure prevent overload by"
    max_new_tokens = 16
    ref_a = baseline.generate_greedy(prompt_a, max_new_tokens=max_new_tokens, stop_on_eos=False)
    ref_b = baseline.generate_greedy(prompt_b, max_new_tokens=max_new_tokens, stop_on_eos=False)

    s1 = Stage1Partition(cfg["model"]["name"], split_layer=3)
    s2 = Stage2Partition(cfg["model"]["name"], split_layer=3)

    inp_a = baseline.tokenizer(prompt_a, return_tensors="pt")["input_ids"]
    inp_b = baseline.tokenizer(prompt_b, return_tensors="pt")["input_ids"]
    gen_a, gen_b = inp_a[0].tolist(), inp_b[0].tolist()

    # Interleave A (5 steps) and B (5 steps)
    tok_a = s2.prefill("req-a", "att-1", s1.prefill("req-a", "att-1", inp_a), selection_mode="worker")
    tok_b = s2.prefill("req-b", "att-1", s1.prefill("req-b", "att-1", inp_b), selection_mode="worker")
    gen_a.append(tok_a)
    gen_b.append(tok_b)
    for _ in range(4):
        tok_a = s2.decode("req-a", "att-1", s1.decode("req-a", "att-1", tok_a), selection_mode="worker")
        tok_b = s2.decode("req-b", "att-1", s1.decode("req-b", "att-1", tok_b), selection_mode="worker")
        gen_a.append(tok_a)
        gen_b.append(tok_b)

    # Reset ONLY req-a and replay on att-2 while req-b continues undisturbed
    assert s1.reset("req-a", "att-1") == 1
    assert s2.reset("req-a", "att-1") == 1
    assert s1.get_cache_length("req-a", "att-1") == 0

    replay_inp_a = torch.tensor([gen_a], dtype=torch.long)
    tok_a = s2.prefill("req-a", "att-2", s1.prefill("req-a", "att-2", replay_inp_a), selection_mode="worker")
    gen_a.append(tok_a)
    tok_b = s2.decode("req-b", "att-1", s1.decode("req-b", "att-1", tok_b), selection_mode="worker")
    gen_b.append(tok_b)

    for _ in range(6, max_new_tokens):
        tok_a = s2.decode("req-a", "att-2", s1.decode("req-a", "att-2", tok_a), selection_mode="worker")
        tok_b = s2.decode("req-b", "att-1", s1.decode("req-b", "att-1", tok_b), selection_mode="worker")
        gen_a.append(tok_a)
        gen_b.append(tok_b)

    assert gen_a == ref_a["generated_ids"]
    assert gen_b == ref_b["generated_ids"]

    s1.clear()
    s2.clear()
    assert s1.active_entries() == 0 and s2.active_entries() == 0


def test_ttl_eviction_and_validation_errors(cfg):
    with pytest.raises(ValueError, match="split_layer"):
        Stage1Partition(cfg["model"]["name"], split_layer=0)
    with pytest.raises(ValueError, match="split_layer"):
        Stage2Partition(cfg["model"]["name"], split_layer=6)

    s1 = Stage1Partition(cfg["model"]["name"], split_layer=3, ttl_seconds=0.05)
    s2 = Stage2Partition(cfg["model"]["name"], split_layer=3, ttl_seconds=0.05)

    inp = torch.tensor([[10, 20, 30]], dtype=torch.long)
    h = s1.prefill("ttl-req", "att1", inp)
    _ = s2.prefill("ttl-req", "att1", h)
    assert s1.active_entries() == 1 and s2.active_entries() == 1

    with pytest.raises(ValueError, match="Cache mismatch"):
        s1.decode("ttl-req", "att1", 40, expected_cache_pos=99)
    with pytest.raises(ValueError, match="selection_mode"):
        s2.decode("ttl-req", "att1", h, selection_mode="invalid")
    with pytest.raises(ValueError, match="non-empty"):
        s1.prefill("", "att1", inp)

    time.sleep(0.08)
    assert s1.evict_expired() == 1
    assert s2.evict_expired() == 1
    assert s1.active_entries() == 0 and s2.active_entries() == 0


def test_proto_and_serialization_roundtrip():
    tensor = torch.randn(1, 4, 768, dtype=torch.float32)
    raw, shape, dtype = tensor_to_bytes(tensor)
    req = inference_pb2.Stage2Request(
        request_id="r1",
        attempt_id="a1",
        step_id=0,
        worker_epoch=1,
        tensor_shape=shape,
        dtype=dtype,
        activation_data=raw,
        cache_position=0,
        selection_mode=inference_pb2.MODE_COORDINATOR_LOGITS,
    )
    reconstructed = bytes_to_tensor(req.activation_data, list(req.tensor_shape), req.dtype)
    assert torch.equal(tensor, reconstructed)
    assert req.ByteSize() > len(raw)