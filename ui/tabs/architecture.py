import streamlit as st

FLOW_DIAGRAM = """
digraph {
    rankdir=LR;
    bgcolor="transparent";
    node [shape=box, style="rounded,filled", fillcolor="#161b22", fontcolor="#e6edf3", color="#2dd4bf", fontname="Helvetica"];
    edge [color="#2dd4bf"];
    fetch [label="Fetch (PDF)"];
    chunk [label="Chunk (basic | advanced)"];
    embed [label="Embed (dense + sparse)"];
    index [label="Index (Qdrant)"];
    retrieve [label="Retrieve (hybrid + RRF)"];
    generate [label="Generate (LLM, cited)"];
    fetch -> chunk -> embed -> index -> retrieve -> generate;
}
"""

STAGES = [
    {
        "title": "1. Fetch",
        "high_level": (
            "Pull the filing PDF from SEC EDGAR and cache it locally so ingestion is "
            "repeatable and offline."
        ),
        "low_level": (
            "`ingestion/fetch_filing_pdf.py: fetch_filing_pdf()` downloads Apple's 10-K PDF "
            "into `data/raw_pdf/AAPL.pdf`. `ingestion/fetch_filings.py` holds the `COMPANIES` "
            "metadata (ticker, CIK, name) used across ingestion."
        ),
    },
    {
        "title": "2. Chunk",
        "high_level": (
            "Split the filing into retrievable passages. Two competing strategies exist so "
            "the demo can compare naive vs. structure-aware chunking."
        ),
        "low_level": (
            "`ingestion/chunk_filings_basic.py`: PyPDFLoader extracts raw text, a regex finds "
            "Item boundaries, then `RecursiveCharacterTextSplitter` splits with "
            "`chunk_size=800`, `chunk_overlap=150` tokens.\n\n"
            "`ingestion/chunk_filings_advanced.py`: docling's `HybridChunker` (with an "
            "`OpenAITokenizer`) parses the PDF's real structure (tables, headings) and "
            "produces table-aware chunks.\n\n"
            "Both tag each chunk with a `pipeline` field (`FilingChunk.pipeline` in "
            "`core/models.py`) so the two runs never mix in Qdrant."
        ),
    },
    {
        "title": "3. Embed",
        "high_level": (
            "Turn each chunk into vectors for both semantic (dense) and keyword (sparse) "
            "search, so hybrid retrieval can use either signal."
        ),
        "low_level": (
            "`core/embeddings.py: embed_dense()` calls OpenAI's `text-embedding-3-small` "
            "(`EMBEDDING_DIM = 1536`). The sparse side is Qdrant's built-in BM25 FastEmbed "
            "model, invoked inline during indexing."
        ),
    },
    {
        "title": "4. Index",
        "high_level": (
            "Store every chunk's dense vector, sparse vector, and metadata (ticker, section, "
            "fiscal year) in Qdrant, ready for retrieval."
        ),
        "low_level": (
            "`ingestion/index_to_qdrant.py: index_all_filings()` upserts into one Qdrant "
            "collection per pipeline -- `sec_filings_basic` and `sec_filings_advanced` -- so "
            "the Pipeline Comparison tab can query either independently."
        ),
    },
    {
        "title": "5. Retrieve",
        "high_level": (
            "Given a question, fetch the most relevant chunks by combining dense and sparse "
            "search. Five interchangeable strategies let the demo show how each affects "
            "answer quality."
        ),
        "low_level": (
            "`core/retrieval.py: hybrid_search()` runs dense + sparse search in one Qdrant "
            "query, fused with Reciprocal Rank Fusion (`Fusion.RRF`), `top_k=8`.\n\n"
            "`core/retrieval_strategies.py` builds on top of it: `baseline` (hybrid_search "
            "directly), `query_rewrite` (LLM rewrites the question first), `hyde` (LLM drafts "
            "a hypothetical answer, embeds that instead), `multi_query` (LLM generates "
            "variants, RRF-fuses their results), `rerank` (over-fetches `top_k*3` candidates, "
            "re-scores with cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`)."
        ),
    },
    {
        "title": "6. Generate",
        "high_level": (
            "Produce a grounded, cited answer -- the model is instructed to answer only from "
            "the retrieved chunks and cite which ones it used."
        ),
        "low_level": (
            "`core/rag.py: answer_question()` builds a prompt constrained to the retrieved "
            "chunks, calls `core/llm.py: complete()` (OpenAI `gpt-4o`), then parses the "
            "model's cited chunk indices back into `Citation` objects. If no chunk is "
            "retrieved, it returns a fixed \"not found\" answer instead of letting the model "
            "guess."
        ),
    },
]

FILE_MAP = [
    ("core/__init__.py", "Loads .env on import so API keys are available everywhere."),
    ("core/models.py", "Shared dataclasses: FilingChunk, RetrievedChunk, Citation, RAGAnswer."),
    ("core/embeddings.py", "Dense embedding calls (OpenAI text-embedding-3-small)."),
    ("core/llm.py", "LLM completion calls (OpenAI gpt-4o)."),
    ("core/retrieval.py", "hybrid_search() -- dense + sparse fusion (RRF) over Qdrant."),
    ("core/retrieval_strategies.py", "Five retrieval strategies built on hybrid_search()."),
    ("core/rag.py", "answer_question() -- the single public entrypoint: retrieve, prompt, cite."),
    ("ingestion/fetch_filing_pdf.py", "Downloads and caches the source PDF."),
    ("ingestion/fetch_filings.py", "Company/ticker/CIK metadata shared across ingestion."),
    ("ingestion/chunk_filings_basic.py", "PyPDFLoader + regex + recursive splitter chunking."),
    ("ingestion/chunk_filings_advanced.py", "docling HybridChunker, table-aware chunking."),
    ("ingestion/index_to_qdrant.py", "Embeds and upserts chunks into a Qdrant collection."),
    ("ingestion/index_comparison_collections.py", "Runs both pipelines into their own collections."),
]


def render() -> None:
    st.caption(
        "How a question turns into a cited answer -- from PDF to response. "
        "Expand each stage for the underlying implementation."
    )
    st.graphviz_chart(FLOW_DIAGRAM, width="stretch")

    for stage in STAGES:
        with st.expander(stage["title"]):
            st.markdown(stage["high_level"])
            st.divider()
            st.markdown(stage["low_level"])

    st.divider()
    st.subheader("File map")
    st.dataframe(
        [{"File": f, "Responsibility": r} for f, r in FILE_MAP],
        width="stretch",
        hide_index=True,
    )
