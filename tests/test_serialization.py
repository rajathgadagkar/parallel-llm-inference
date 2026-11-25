import pytest
import torch
from common.serialization import bytes_to_tensor, tensor_to_bytes


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_roundtrip_dtypes(dtype):
    if dtype == torch.int64:
        t = torch.randint(0, 100, (1, 4, 8), dtype=dtype)
    else:
        t = torch.randn(1, 4, 8, dtype=dtype)
    raw, shape, dtype_str = tensor_to_bytes(t)
    out = bytes_to_tensor(raw, shape, dtype_str)
    assert torch.equal(t, out)
    assert out.dtype == dtype


def test_non_contiguous_tensor_roundtrip():
    base = torch.randn(2, 5, 3, dtype=torch.float32)
    non_contig = base.transpose(0, 2)
    assert not non_contig.is_contiguous()
    raw, shape, dtype_str = tensor_to_bytes(non_contig)
    out = bytes_to_tensor(raw, shape, dtype_str)
    assert torch.equal(non_contig, out)


def test_unsupported_dtype_raises():
    t = torch.randn(2, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="Unsupported tensor dtype"):
        tensor_to_bytes(t)

    with pytest.raises(ValueError, match="Unsupported dtype string"):
        bytes_to_tensor(b"\x00" * 16, [2, 2], "float64")


def test_negative_shape_raises():
    with pytest.raises(ValueError, match="negative dimensions"):
        bytes_to_tensor(b"\x00" * 16, [1, -2, 2], "float32")


def test_byte_length_mismatch_raises():
    with pytest.raises(ValueError, match="Byte length mismatch"):
        bytes_to_tensor(b"\x00" * 8, [1, 4], "float32")