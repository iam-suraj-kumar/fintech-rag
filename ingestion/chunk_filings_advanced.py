import json
from pathlib import Path

import tiktoken
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.transforms.serializer.markdown import MarkdownParams, MarkdownTableSerializer

from core.models import FilingChunk
from ingestion.fetch_filings import COMPANIES

RAW_PDF_DIR = Path("data/raw_pdf")
CHUNKS_DIR = Path("data/chunks_advanced")


class _MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """Serializes tables as markdown instead of flattened text, so chunks keep table structure."""

    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=True),
        )


def _build_chunker() -> HybridChunker:
    tokenizer = OpenAITokenizer(tokenizer=tiktoken.encoding_for_model("gpt-4o"), max_tokens=800)
    return HybridChunker(
        tokenizer=tokenizer,
        serializer_provider=_MarkdownTableSerializerProvider(),
    )


def chunk_filing_advanced(ticker: str, pdf_path: Path, fiscal_year: str) -> list[FilingChunk]:
    company_name = COMPANIES[ticker]["name"]
    document = DocumentConverter().convert(str(pdf_path)).document
    chunker = _build_chunker()

    chunks = []
    for chunk in chunker.chunk(dl_doc=document):
        text = chunker.contextualize(chunk)
        if not text.strip():
            continue
        section = " > ".join(chunk.meta.headings) if chunk.meta.headings else "Full Document"
        chunks.append(
            FilingChunk(
                ticker=ticker,
                company_name=company_name,
                filing_type="10-K",
                section=section,
                fiscal_year=fiscal_year,
                text=text,
                pipeline="advanced",
            )
        )
    return chunks


def chunk_all_filings_advanced(fiscal_year: str = "2024") -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in COMPANIES:
        pdf_path = RAW_PDF_DIR / f"{ticker}.pdf"
        chunks = chunk_filing_advanced(ticker, pdf_path, fiscal_year)
        out_path = CHUNKS_DIR / f"{ticker}.json"
        out_path.write_text(
            json.dumps([c.__dict__ for c in chunks], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    chunk_all_filings_advanced()
