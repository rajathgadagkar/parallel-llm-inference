"""Compile proto/inference.proto into proto/generated/ with package-relative imports."""
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(REPO_ROOT / "proto"),
        f"--python_out={REPO_ROOT / 'proto' / 'generated'}",
        f"--grpc_python_out={REPO_ROOT / 'proto' / 'generated'}",
        str(REPO_ROOT / "proto" / "inference.proto"),
    ]
    subprocess.run(cmd, check=True)

    grpc_file = REPO_ROOT / "proto" / "generated" / "inference_pb2_grpc.py"
    text = grpc_file.read_text(encoding="utf-8")
    updated = text.replace(
        "import inference_pb2 as inference__pb2",
        "from . import inference_pb2 as inference__pb2",
        1,
    )
    grpc_file.write_text(updated, encoding="utf-8")
    print("Regenerated proto/generated/inference_pb2.py and inference_pb2_grpc.py")


if __name__ == "__main__":
    main()
