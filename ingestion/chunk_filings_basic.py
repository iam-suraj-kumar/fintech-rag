import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.models import FilingChunk
from ingestion.chunk_filings import _token_length, split_into_sections
from ingestion.fetch_filings import COMPANIES

RAW_PDF_DIR = Path("data/raw_pdf")
CHUNKS_DIR = Path("data/chunks_basic")


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text with LangChain's built-in PyPDFLoader -- naive per-page text, no layout awareness."""
    pages = PyPDFLoader(str(pdf_path)).load()
    return "\n".join(page.page_content for page in pages)


def chunk_filing_basic(ticker: str, pdf_path: Path, fiscal_year: str) -> list[FilingChunk]:
    company_name = COMPANIES[ticker]["name"]
    text = pdf_to_text(pdf_path)
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
                    pipeline="basic",
                )
            )
    return chunks


def chunk_all_filings_basic(fiscal_year: str = "2024") -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in COMPANIES:
        pdf_path = RAW_PDF_DIR / f"{ticker}.pdf"
        chunks = chunk_filing_basic(ticker, pdf_path, fiscal_year)
        out_path = CHUNKS_DIR / f"{ticker}.json"
        out_path.write_text(
            json.dumps([c.__dict__ for c in chunks], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    chunk_all_filings_basic()
