from pathlib import Path
import yaml


def load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        if candidate.exists():
            path = candidate

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Config file {path} must contain a non-empty YAML mapping")

    return data