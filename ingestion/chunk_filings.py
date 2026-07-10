import json
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.models import FilingChunk
from ingestion.fetch_filings import COMPANIES

RAW_DIR = Path("data/raw")
CHUNKS_DIR = Path("data/chunks")

ITEM_HEADER_RE = re.compile(
    r"^(Item\s+\d+[A-Z]?\.?\s+[A-Za-z][^\n]{0,80})", re.IGNORECASE | re.MULTILINE
)

_encoding = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    return len(_encoding.encode(text))


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_title, section_text) pairs using Item headers.

    Falls back to a single "Full Document" section if no Item headers are found,
    since SEC filers format Item headers inconsistently and ingestion must not fail.
    """
    matches = list(ITEM_HEADER_RE.finditer(text))
    if not matches:
        return [("Full Document", text)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(1).strip()
        body = text[start:end]
        sections.append((title, body))
    return sections


def chunk_filing(ticker: str, html: str, fiscal_year: str) -> list[FilingChunk]:
    company_name = COMPANIES[ticker]["name"]
    text = html_to_text(html)
    sections = split_into_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, length_function=_token_length
    )
    chunks = []
    for section_title, section_text in sections:
        for piece in splitter.split_text(section_text):
            if not piece.strip():
                continue
            chunks.append(
                FilingChunk(
                    ticker=ticker,
                    company_name=company_name,
                    filing_type="10-K",
                    section=section_title,
                    fiscal_year=fiscal_year,
                    text=piece,
                )
            )
    return chunks


def chunk_all_filings(fiscal_year: str = "2024") -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in COMPANIES:
        raw_path = RAW_DIR / f"{ticker}.html"
        html = raw_path.read_text(encoding="utf-8")
        chunks = chunk_filing(ticker, html, fiscal_year)
        out_path = CHUNKS_DIR / f"{ticker}.json"
        out_path.write_text(
            json.dumps([c.__dict__ for c in chunks], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    chunk_all_filings()
