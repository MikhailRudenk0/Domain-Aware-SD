from pathlib import Path
from unittest.mock import patch

import pytest

from src.eval.model_loader import resolve_model_dir


def test_local_dir_short_circuits(tmp_path: Path):
    """If the model is already populated locally, no S3 client should be created."""
    (tmp_path / "my-model").mkdir()
    (tmp_path / "my-model" / "config.json").write_text("{}")

    with patch("src.eval.model_loader.make_client") as mock_make_client:
        result = resolve_model_dir(
            "my-model", project_root=tmp_path, s3_client=None
        )

    assert result == tmp_path / "my-model"
    mock_make_client.assert_not_called()


def test_empty_local_dir_triggers_s3_lookup(tmp_path: Path):
    """An empty directory should not count as populated."""
    (tmp_path / "my-model").mkdir()  # empty

    fake_client = object()
    with patch("src.eval.model_loader.list_prefix", return_value=[]) as mock_list:
        with pytest.raises(FileNotFoundError):
            resolve_model_dir(
                "my-model", project_root=tmp_path, s3_client=fake_client
            )

    mock_list.assert_called_once_with(fake_client, "models/my-model")


def test_missing_locally_and_on_s3_raises(tmp_path: Path):
    fake_client = object()
    with patch("src.eval.model_loader.list_prefix", return_value=[]):
        with pytest.raises(FileNotFoundError, match="not present at"):
            resolve_model_dir(
                "absent-model", project_root=tmp_path, s3_client=fake_client
            )


def test_s3_download_invoked_when_local_missing(tmp_path: Path):
    """When local is missing and S3 has files, download_prefix is called."""
    fake_client = object()
    objects = [{"Key": "models/my-model/config.json", "Size": 10}]

    def fake_download_prefix(client, prefix, local_dir, force, workers, label):
        # Simulate successful download
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    with patch("src.eval.model_loader.list_prefix", return_value=objects), \
         patch("src.eval.model_loader.download_prefix",
               side_effect=fake_download_prefix) as mock_dl:
        result = resolve_model_dir(
            "my-model", project_root=tmp_path, s3_client=fake_client
        )

    assert result == tmp_path / "my-model"
    assert (result / "config.json").exists()
    mock_dl.assert_called_once()


def test_s3_download_producing_nothing_raises(tmp_path: Path):
    """If list_prefix returned objects but download_prefix silently failed to
    create files, we should raise FileNotFoundError."""
    fake_client = object()
    objects = [{"Key": "models/my-model/config.json", "Size": 10}]

    with patch("src.eval.model_loader.list_prefix", return_value=objects), \
         patch("src.eval.model_loader.download_prefix"):
        with pytest.raises(FileNotFoundError, match="produced no files"):
            resolve_model_dir(
                "my-model", project_root=tmp_path, s3_client=fake_client
            )
