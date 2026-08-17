"""Static guardrails for the clean-clone Docker deployment path.

This check intentionally performs no network or Docker operations.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _dependency_names() -> set[str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        item.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].split("<", 1)[0].strip().lower()
        for item in pyproject["project"]["dependencies"]
    }


def main() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    dependencies = _dependency_names()

    assert "COPY requirements/" not in dockerfile
    assert "uv pip install --system --no-cache ." in dockerfile
    assert 'COPY agent/ ./agent/' in dockerfile
    assert 'COPY common/ ./common/' in dockerfile
    assert 'COPY ingestion/ ./ingestion/' in dockerfile
    assert "requirements/" in dockerignore
    assert "aio-pika" in dependencies
    assert "langchain-text-splitters" in dependencies
    assert compose.count("image: agentic-rag-project-main2-runtime:latest") == 2
    assert "target: runtime" in compose
    assert "PYPI_INDEX_URL:" in compose
    assert "NPM_REGISTRY:" in compose
    print("first-deploy-contract: ok")


if __name__ == "__main__":
    main()
