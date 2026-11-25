import math
import numpy as np
import torch

_DTYPE_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
}

_TORCH_TO_STR = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.int64: "int64",
}


def tensor_to_bytes(tensor: torch.Tensor) -> tuple[bytes, list[int], str]:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype not in _TORCH_TO_STR:
        raise ValueError(
            f"Unsupported tensor dtype '{tensor.dtype}'. Supported: {list(_TORCH_TO_STR.keys())}"
        )
    t = tensor.detach().cpu().contiguous()
    dtype_str = _TORCH_TO_STR[t.dtype]
    return t.numpy().tobytes(), list(t.shape), dtype_str


def bytes_to_tensor(raw_bytes: bytes, shape: list[int], dtype_str: str) -> torch.Tensor:
    if dtype_str not in _DTYPE_TO_NUMPY:
        raise ValueError(
            f"Unsupported dtype string '{dtype_str}'. Supported: {list(_DTYPE_TO_NUMPY.keys())}"
        )
    if any(dim < 0 for dim in shape):
        raise ValueError(f"Tensor shape cannot contain negative dimensions: {shape}")

    np_dtype = _DTYPE_TO_NUMPY[dtype_str]
    expected_bytes = math.prod(shape) * np.dtype(np_dtype).itemsize
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            f"Byte length mismatch: got {len(raw_bytes)} bytes, expected {expected_bytes} "
            f"for shape={shape} and dtype='{dtype_str}'"
        )

    arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)