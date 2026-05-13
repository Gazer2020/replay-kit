from __future__ import annotations

from pathlib import Path


def test_gitignore_protects_large_artifacts() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in [".env", "outputs/runs/", "*.pth", "*.pt", "*.ckpt", "*.safetensors"]:
        assert pattern in text
    assert "!data/*/README.md" in text
