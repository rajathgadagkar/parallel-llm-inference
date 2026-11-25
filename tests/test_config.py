import pytest
from common.config import load_config


def test_load_valid_config():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, dict)
    assert cfg["model"]["split_layer"] == 3


def test_load_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_empty_yaml_raises(tmp_path):
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty YAML mapping"):
        load_config(empty_file)


def test_load_non_dict_yaml_raises(tmp_path):
    list_file = tmp_path / "list.yaml"
    list_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty YAML mapping"):
        load_config(list_file)