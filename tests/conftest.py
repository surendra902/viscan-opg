import pytest

import app as app_module


@pytest.fixture(autouse=True)
def _isolate_results(tmp_path, monkeypatch):
    """Keep test artifacts out of the app's real results directory."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    monkeypatch.setattr(app_module, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(app_module.service, "results_dir", results_dir)
