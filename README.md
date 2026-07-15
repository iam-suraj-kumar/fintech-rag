# FinTech RAG

A retrieval-augmented generation demo that answers questions about public companies' SEC 10-K filings, with citations grounded in the source text.

Currently configured for Apple only (`ingestion/fetch_filings.py:COMPANIES`); add more tickers there to extend it. The filing is ingested as a PDF through two parallel chunking pipelines — "basic" (naive PDF text extraction) and "advanced" (docling layout-aware parsing) — each embedded and indexed into its own Qdrant collection so they can be compared side by side. Questions are answered by an LLM constrained to cite only the retrieved excerpts.

## Architecture

```
ingestion/fetch_filing_pdf.py       → data/raw_pdf/{TICKER}.pdf         (company-published 10-K PDF)
ingestion/chunk_filings_basic.py    → data/chunks_basic/{TICKER}.json    (PyPDFLoader + regex sections)
ingestion/chunk_filings_advanced.py → data/chunks_advanced/{TICKER}.json (docling HybridChunker, layout-aware)
ingestion/text_utils.py             — shared token-length + Item-header section splitting
ingestion/index_to_qdrant.py        → Qdrant collection (dense + sparse vectors); also used by:
ingestion/index_comparison_collections.py → indexes both chunk sets into
                                       "sec_filings_basic" and "sec_filings_advanced"

core/retrieval.py  hybrid_search()   — dense + sparse fusion (RRF) over Qdrant
core/rag.py        answer_question() — retrieves chunks, prompts LLM, parses cited answer
core/llm.py        complete()        — chat completion via Portkey, with token/cost accounting
core/embeddings.py embed_dense()     — dense embeddings via Portkey
core/clients.py                      — Qdrant / sparse-model client factories; COLLECTION_NAME sets
                                        which collection the Q&A tab queries (currently "sec_filings_advanced")
core/models.py                       — FilingChunk, RetrievedChunk, Citation, RAGAnswer

ui/app.py                     — Streamlit app entrypoint (Q&A, Pipeline Comparison, Architecture, Eval tabs)
ui/tabs/qa.py                  — chat UI over answer_question()
ui/tabs/pipeline_comparison.py — side-by-side basic vs. advanced retrieval/chunking comparison
ui/tabs/evaluation.py          — runs/inspects the eval/ golden-dataset harness

eval/golden_dataset.py — hand-labeled questions with expected sections/answers, used to score retrieval + generation
eval/run_eval.py        — scores answer_question() against the golden set
```

Console logging: ingestion scripts and `core/retrieval.py`/`core/rag.py` emit `logging` INFO records (chunk counts, upsert counts, retrieval hits, token usage). `ui/app.py` and each ingestion script's `__main__` block call `logging.basicConfig(level=logging.INFO)`, so logs show up in the terminal running Streamlit or the script.

## Setup

Requires Python 3.11+. Qdrant runs in embedded "local mode" by default (on-disk storage under `data/qdrant_local/`) — no server or Docker required.

```bash
pip install -e ".[dev]"
cp .env.example .env          # fill in API keys
```

`.env` configuration:

| Variable | Purpose |
|---|---|
| `PORTKEY_API_KEY` | required — both LLM completion (`core/llm.py`) and dense embeddings (`core/embeddings.py`) call OpenAI-family models through Portkey |
| `QDRANT_LOCAL_PATH` | optional override, default `data/qdrant_local` — on-disk path for Qdrant's embedded local mode |
| `QDRANT_URL` | optional — set to point at a running Qdrant server instead of local mode (e.g. `docker compose up -d` with `docker-compose.yml`, then `http://localhost:6333`) |
| `LLM_MODEL` | optional override, default `@demo-fintech/gpt-4o` |
| `EMBEDDING_MODEL` | optional override, default `@demo-fintech/text-embedding-3-small` |

## Ingestion pipeline

Run in order to populate both comparison collections from scratch:

```bash
python -m ingestion.fetch_filing_pdf              # fetch each ticker's 10-K PDF into data/raw_pdf/
python -m ingestion.chunk_filings_basic            # naive PyPDFLoader + regex sections → data/chunks_basic/
python -m ingestion.chunk_filings_advanced         # docling layout-aware chunking → data/chunks_advanced/
python -m ingestion.index_comparison_collections   # embed + upsert both sets into
                                                    #   "sec_filings_basic" and "sec_filings_advanced"
```

`core/clients.py:COLLECTION_NAME` controls which collection the Q&A tab and `answer_question()` query by default (currently `"sec_filings_advanced"`); pass `collection_name=` explicitly to query a different one. Each step logs progress (pages/sections/chunks/points) via `logging` — see the Architecture section above.

## Usage

```bash
python scripts/smoke_test.py   # sanity-check the RAG pipeline with sample questions
streamlit run ui/app.py        # interactive Q&A UI, Pipeline Comparison, Architecture, and Eval tabs
python -m eval.run_eval         # score answer_question() against eval/golden_dataset.py
```

Or call the pipeline directly:

```python
from core.rag import answer_question

result = answer_question("How does Apple describe supply chain risk?")
print(result.answer)
for citation in result.citations:
    print(citation.company_name, citation.section, citation.fiscal_year)
```

## Sample questions

Questions to try against the demo corpus, spanning exact-term and semantic retrieval. Note: `PDF_URLS` in `ingestion/fetch_filing_pdf.py` points at Apple's *latest* published 10-K PDF, which drifts over time as Apple files new ones — chunks are tagged `fiscal_year="2024"` by a hardcoded default in the chunking scripts regardless of which fiscal year the PDF actually covers, so double-check that label against the retrieved text before trusting a "fiscal 2024" answer.

**Exact-term / factual**
- What was Apple's total net sales for fiscal year 2024?
- What was Apple's gross margin percentage in 2024 vs. 2023?
- How much did Apple spend on R&D in fiscal 2024, and what percent of net sales is that?
- What are the interest rates and maturity dates on Apple's outstanding notes?

**Segment / numerical reasoning**
- Which geographic segment had the largest revenue decline in 2024, and why?
- How did Services net sales compare to iPhone net sales in 2024?
- Why did Greater China net sales decrease in fiscal 2024?
- Compare Products gross margin to Services gross margin — which is higher and why?

**Semantic / conceptual**
- How does Apple describe the risks of "Apple Intelligence" and generative AI in its filing?
- What cybersecurity risks does Apple disclose in Item 1C?
- What does Apple say about risks from foreign currency fluctuation?
- What legal proceedings is Apple currently involved in?

**Cross-section synthesis**
- What new products did Apple launch in fiscal 2024, and how did they affect segment performance?
- What macroeconomic or geopolitical risks does Apple flag that could affect its Greater China revenue?

**"Not found" trap** (should trigger the grounded-refusal path)
- What was Apple's revenue in fiscal year 2019?
- What is Apple's stock buyback plan for fiscal 2026?

## Tests

```bash
pytest
```
