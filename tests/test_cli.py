from __future__ import annotations

import sys
import os
from pathlib import Path
from types import SimpleNamespace

from deck_vision.cli import main


def test_cli_missing_image_reports_structured_error(capsys) -> None:
    code = main(["recognize", "missing.png", "--json"])

    captured = capsys.readouterr()
    assert code == 2
    assert "image_not_found" in captured.err


def test_cli_serve_runs_uvicorn(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_run(app: str, **kwargs: object) -> None:
        calls.append((app, kwargs))

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=fake_run))
    monkeypatch.delenv("ASSETS_API_ENDPOINT", raising=False)
    monkeypatch.delenv("DECK_VISION_CACHE_DIR", raising=False)

    code = main(
        [
            "--endpoint",
            "https://example.test/api",
            "--cache-dir",
            str(tmp_path),
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--reload",
            "--workers",
            "2",
        ]
    )

    assert code == 0
    assert calls == [
        (
            "deck_vision.server:app",
            {"host": "0.0.0.0", "port": 9000, "reload": True, "workers": 2},
        )
    ]
    assert os.environ.get("ASSETS_API_ENDPOINT") is None
    assert os.environ.get("DECK_VISION_CACHE_DIR") is None
