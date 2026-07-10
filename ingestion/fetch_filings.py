import time
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
USER_AGENT = "FinTech RAG Demo contact@example.com"

COMPANIES = {
    "AAPL": {"cik": "0000320193", "name": "Apple Inc."},
    "JPM": {"cik": "0000019617", "name": "JPMorgan Chase & Co."},
    "V": {"cik": "0001403161", "name": "Visa Inc."},
    "BAC": {"cik": "0000070858", "name": "Bank of America Corporation"},
    "WFC": {"cik": "0000072971", "name": "Wells Fargo & Company"},
}


def _headers() -> dict:
    return {"User-Agent": USER_AGENT}


def get_latest_10k_url(cik: str) -> str:
    """Return the URL of the most recent 10-K filing document for a given CIK."""
    resp = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            cik_no_zeros = str(int(cik))
            return (
                f"https://www.sec.gov/Archives/edgar/data/"
                f"{cik_no_zeros}/{accession}/{primary_doc}"
            )
    raise ValueError(f"No 10-K filing found for CIK {cik}")


def fetch_filing_html(url: str) -> str:
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_all_filings(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for ticker, info in COMPANIES.items():
        out_path = RAW_DIR / f"{ticker}.html"
        if out_path.exists() and not force:
            continue
        url = get_latest_10k_url(info["cik"])
        html = fetch_filing_html(url)
        out_path.write_text(html, encoding="utf-8")
        time.sleep(0.15)  # stay well under EDGAR's 10 req/sec rate limit


if __name__ == "__main__":
    fetch_all_filings()
