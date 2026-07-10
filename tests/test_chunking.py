SYNTHETIC_HTML = """
<html><body>
<p>Item 1. Business</p>
<p>We design, manufacture and market synthetic widgets for testing purposes. """ + (
    "Widget details. " * 200
) + """</p>
<p>Item 7. Management's Discussion and Analysis</p>
<p>Revenue grew due to strong widget demand. """ + (
    "Financial detail. " * 200
) + """</p>
</body></html>
"""


def test_split_into_sections_finds_item_headers():
    from ingestion.chunk_filings import html_to_text, split_into_sections

    text = html_to_text(SYNTHETIC_HTML)
    sections = split_into_sections(text)
    titles = [title for title, _ in sections]

    assert any(t.startswith("Item 1.") for t in titles)
    assert any(t.startswith("Item 7.") for t in titles)


def test_split_into_sections_falls_back_when_no_items_found():
    from ingestion.chunk_filings import split_into_sections

    sections = split_into_sections("Just plain filing text with no item headers at all.")

    assert sections == [
        ("Full Document", "Just plain filing text with no item headers at all.")
    ]


def test_chunk_filing_tags_chunks_with_correct_section_and_metadata(monkeypatch):
    import ingestion.chunk_filings as mod

    monkeypatch.setitem(mod.COMPANIES, "TEST", {"cik": "0000000000", "name": "Test Corp"})

    chunks = mod.chunk_filing("TEST", SYNTHETIC_HTML, fiscal_year="2024")

    assert len(chunks) > 0
    assert all(c.ticker == "TEST" for c in chunks)
    assert all(c.fiscal_year == "2024" for c in chunks)

    item1_chunks = [c for c in chunks if c.section.startswith("Item 1.")]
    item7_chunks = [c for c in chunks if c.section.startswith("Item 7.")]
    assert len(item1_chunks) > 0
    assert len(item7_chunks) > 0
    assert "widget" in item1_chunks[0].text.lower()
