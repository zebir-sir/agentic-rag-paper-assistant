import pytest

from agent.ingestion_jobs import _download_binary_file
from agent.tools import _extract_pdf_url


def test_openalex_pdf_policy_does_not_treat_landing_page_as_pdf():
    work = {
        "open_access": {"is_oa": True, "oa_url": "https://doi.org/10.1109/example"},
        "best_oa_location": {"landing_page_url": "https://publisher.example/paper"},
        "primary_location": {"landing_page_url": "https://doi.org/10.1109/example"},
    }
    assert _extract_pdf_url(work) is None


def test_openalex_pdf_policy_keeps_explicit_pdf_url():
    work = {"best_oa_location": {"pdf_url": "https://repository.example/paper.pdf"}}
    assert _extract_pdf_url(work) == "https://repository.example/paper.pdf"


def test_openalex_download_rejects_html_response(monkeypatch):
    class FakeResponse:
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"<html>publisher landing page</html>"

    monkeypatch.setattr("agent.ingestion_jobs.urlopen", lambda *_args, **_kwargs: FakeResponse())
    with pytest.raises(ValueError, match="did not return a PDF"):
        _download_binary_file("https://publisher.example/paper", object())


def test_openalex_download_accepts_pdf_signature(monkeypatch):
    class FakeResponse:
        headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"%PDF-1.7\nmock pdf"

    class MemoryTarget:
        data = b""

        def write_bytes(self, data):
            self.data = data

    monkeypatch.setattr("agent.ingestion_jobs.urlopen", lambda *_args, **_kwargs: FakeResponse())
    target = MemoryTarget()
    _download_binary_file("https://repository.example/paper.pdf", target)
    assert target.data.startswith(b"%PDF-")
