from ingestion.title_parser import extract_document_title, filename_title


def test_extract_document_title_prefers_markdown_heading_over_staged_filename():
    content = "# Rapidly-Exploring Random Trees: A New Tool for Path Planning\n\nSteven M. LaValle\n\n## Abstract"

    title = extract_document_title(content, "/tmp/8ab12f01_C01_LaValle1998_RapidlyExploringRandomTrees.pdf")

    assert title == "Rapidly-Exploring Random Trees: A New Tool for Path Planning"


def test_filename_title_removes_upload_staging_prefix():
    assert filename_title("/tmp/8ab12f01_AITstar.pdf") == "AITstar"


def test_extract_document_title_falls_back_for_garbled_embedded_font_text():
    content = "✶✸✷✺✹☛✻✼✹✾✽✝✿❁❀❃❂❅❄❇❆✗❄❉❈❊❈❋✹❇●■❍❑❏▼▲"

    title = extract_document_title(content, "/tmp/2001_Randomized_Kinodynamic_Planning.pdf")

    assert title == "2001 Randomized Kinodynamic Planning"
