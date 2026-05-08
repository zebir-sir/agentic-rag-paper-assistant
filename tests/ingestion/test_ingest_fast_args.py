import argparse
from types import SimpleNamespace

import pytest

from ingestion import ingest as ingest_module


@pytest.mark.asyncio
async def test_fast_flag_disables_semantic_images_tables(monkeypatch):
    captured = {}

    class DummyPipeline:
        def __init__(
            self,
            config,
            documents_folder="documents",
            clean_before_ingest=False,
            reset_kb_before_ingest=False,
            sql_schema_path="sql/schema.sql",
            include_images=True,
            include_tables=True,
        ):
            captured["use_semantic_chunking"] = config.use_semantic_chunking
            captured["include_images"] = include_images
            captured["include_tables"] = include_tables
            captured["reset_kb_before_ingest"] = reset_kb_before_ingest
            self.extractor_config = SimpleNamespace(
                include_images=include_images,
                include_tables=include_tables,
            )

        async def ingest_documents(self, _progress_callback=None):
            return []

        async def close(self):
            return None

    monkeypatch.setattr(ingest_module, "DocumentIngestionPipeline", DummyPipeline)
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            documents="documents",
            clean=False,
            chunk_size=850,
            no_semantic=False,
            chunk_overlap=150,
            verbose=False,
            sql_schema_path="sql/schema.sql",
            no_images=False,
            no_tables=False,
            fast=True,
            reset_kb=False,
        ),
    )

    await ingest_module.main()

    assert captured["use_semantic_chunking"] is False
    assert captured["include_images"] is False
    assert captured["include_tables"] is False
    assert captured["reset_kb_before_ingest"] is False
