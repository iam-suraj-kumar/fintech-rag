import re

from core.llm import complete
from core.models import Citation, RAGAnswer, RetrievedChunk
from core.retrieval import hybrid_search

NOT_FOUND_MESSAGE = "I couldn't find information about this in the available filings."


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = "\n\n".join(
        f"[{i}] ({chunk.company_name}, {chunk.section}, FY{chunk.fiscal_year})\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the numbered filing excerpts below. "
        "After your answer, on a new line, write 'CITED: ' followed by a comma-separated "
        "list of the excerpt numbers you used (e.g. 'CITED: 0,2'). "
        "If the excerpts don't contain enough information to answer, say so explicitly "
        "and write 'CITED: ' with nothing after it.\n\n"
        f"Excerpts:\n{context_blocks}\n\nQuestion: {question}"
    )


def _parse_response(
    raw_text: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    text = raw_text.strip()
    match = re.search(r"CITED:\s*([\d,\s]*)\s*$", text)
    if not match:
        return text, []

    answer_text = text[: match.start()].strip()
    indices_str = match.group(1).strip()
    if not indices_str:
        return answer_text, []

    citations = []
    for idx_str in indices_str.split(","):
        idx_str = idx_str.strip()
        if not idx_str.isdigit():
            continue
        idx = int(idx_str)
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            citations.append(
                Citation(
                    ticker=chunk.ticker,
                    company_name=chunk.company_name,
                    section=chunk.section,
                    fiscal_year=chunk.fiscal_year,
                    text_snippet=chunk.text[:300],
                )
            )
    return answer_text, citations


def answer_question(question: str) -> RAGAnswer:
    """Retrieve relevant filing chunks via hybrid search and generate a grounded, cited answer."""
    chunks = hybrid_search(question, top_k=8)

    if not chunks:
        return RAGAnswer(question=question, answer=NOT_FOUND_MESSAGE, citations=[])

    prompt = _build_prompt(question, chunks)
    raw_text = complete(prompt)
    answer_text, citations = _parse_response(raw_text, chunks)

    if not citations:
        return RAGAnswer(question=question, answer=NOT_FOUND_MESSAGE, citations=[])

    return RAGAnswer(question=question, answer=answer_text, citations=citations)
