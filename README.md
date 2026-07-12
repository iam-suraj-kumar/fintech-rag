# FinTech RAG

A retrieval-augmented generation demo that answers questions about public companies' SEC 10-K filings, with citations grounded in the source text.

Filings for five companies (Apple, JPMorgan, Visa, Bank of America, Wells Fargo) are fetched from SEC EDGAR, chunked by section, embedded, and indexed in Qdrant for hybrid (dense + sparse) retrieval. Questions are answered by an LLM constrained to cite only the retrieved excerpts.

## Architecture

```
ingestion/fetch_filings.py   → data/raw/{TICKER}.html      (SEC EDGAR 10-Ks)
ingestion/chunk_filings.py   → data/chunks/{TICKER}.json    (section-tagged FilingChunks)
ingestion/index_to_qdrant.py → Qdrant "sec_filings" collection (dense + sparse vectors)

core/retrieval.py  hybrid_search()   — dense + sparse fusion (RRF) over Qdrant
core/rag.py        answer_question() — retrieves chunks, prompts LLM, parses cited answer
core/llm.py        complete()        — provider-agnostic LLM call (anthropic | openai | local)
core/embeddings.py embed_dense()     — provider-agnostic dense embeddings (voyage | openai | local)
core/clients.py                      — Qdrant / sparse-model / dense-embedding client factories
core/models.py                       — FilingChunk, RetrievedChunk, Citation, RAGAnswer

ui/app.py — Streamlit chat UI over answer_question()
```

## Setup

Requires Python 3.11+ and Docker (for Qdrant).

```bash
pip install -e ".[dev]"
docker compose up -d          # starts Qdrant on localhost:6333
cp .env.example .env          # fill in API keys
```

`.env` configuration:

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `VOYAGE_API_KEY` | provider credentials |
| `QDRANT_URL` | Qdrant endpoint (default `http://localhost:6333`) |
| `LLM_PROVIDER` | `anthropic` \| `openai` \| `local` |
| `EMBEDDING_PROVIDER` | `voyage` \| `openai` \| `local` |
| `LLM_MODEL` / `EMBEDDING_MODEL` | optional overrides; see `core/llm.py` / `core/embeddings.py` for defaults |
| `LOCAL_LLM_BASE_URL` / `LOCAL_EMBEDDING_BASE_URL` | used only when the corresponding provider is `local` (e.g. Ollama) |

## Ingestion pipeline

Run in order to populate the index from scratch:

```bash
python -m ingestion.fetch_filings     # fetch latest 10-Ks into data/raw/
python -m ingestion.chunk_filings     # split into section-tagged chunks in data/chunks/
python -m ingestion.index_to_qdrant   # embed + upsert chunks into Qdrant
```

## Usage

```bash
python scripts/smoke_test.py   # sanity-check the RAG pipeline with sample questions
streamlit run ui/app.py        # interactive Q&A UI
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

Questions to try against the demo corpus (Apple's FY2024 10-K), spanning exact-term and semantic retrieval:

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
