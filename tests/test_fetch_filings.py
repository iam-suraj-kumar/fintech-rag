import json
from unittest.mock import patch, Mock

SUBMISSIONS_FIXTURE = {
    "filings": {
        "recent": {
            "form": ["8-K", "10-K", "10-Q"],
            "accessionNumber": [
                "0000320193-24-000010",
                "0000320193-24-000005",
                "0000320193-24-000001",
            ],
            "primaryDocument": ["form8k.htm", "aapl-10k.htm", "form10q.htm"],
        }
    }
}


def test_get_latest_10k_url_finds_first_10k_form():
    from ingestion.fetch_filings import get_latest_10k_url

    with patch("ingestion.fetch_filings.requests.get") as mock_get:
        mock_get.return_value = Mock(
            json=lambda: SUBMISSIONS_FIXTURE, raise_for_status=lambda: None
        )
        url = get_latest_10k_url("0000320193")

    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/"
        "000032019324000005/aapl-10k.htm"
    )


def test_fetch_all_filings_skips_existing_cache(tmp_path, monkeypatch):
    import ingestion.fetch_filings as mod

    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    cached = tmp_path / "AAPL.html"
    cached.write_text("cached content")

    with patch.object(mod, "get_latest_10k_url") as mock_url, patch.object(
        mod, "fetch_filing_html"
    ) as mock_fetch, patch.object(mod, "time") as mock_time:
        mock_url.return_value = "http://example.com/x.htm"
        mock_fetch.return_value = "<html>new</html>"
        mod.fetch_all_filings()

    assert cached.read_text() == "cached content"
    assert mock_fetch.call_count == len(mod.COMPANIES) - 1
