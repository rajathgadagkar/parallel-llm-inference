from pathlib import Path
import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        if candidate.exists():
            path = candidate
    with open(path, "r") as f:
        return yaml.safe_load(f)
    