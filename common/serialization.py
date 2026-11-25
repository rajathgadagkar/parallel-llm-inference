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
    t = tensor.detach().cpu().contiguous()
    dtype_str = _TORCH_TO_STR[t.dtype]
    return t.numpy().tobytes(), list(t.shape), dtype_str


def bytes_to_tensor(raw_bytes: bytes, shape: list[int], dtype_str: str) -> torch.Tensor:
    np_dtype = _DTYPE_TO_NUMPY[dtype_str]
    arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape).copy()
    return torch.from_numpy(arr)