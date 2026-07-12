from unittest.mock import patch

from core.models import RetrievedChunk
import core.rag as mod


def _make_chunk(ticker="JPM", text="Net interest margin was 2.7%.", section="Item 7", score=0.02):
    return RetrievedChunk(
        id=0,
        ticker=ticker,
        company_name="JPMorgan Chase & Co.",
        filing_type="10-K",
        section=section,
        fiscal_year="2024",
        text=text,
        score=score,
    )


def test_answer_question_returns_grounded_answer_with_citations():
    chunk = _make_chunk()

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "complete", return_value="The net interest margin was 2.7% for the fiscal year.\nCITED: 0"
    ):
        result = mod.answer_question("What was JPMorgan's net interest margin?")

    assert "2.7%" in result.answer
    assert len(result.citations) == 1
    assert result.citations[0].ticker == "JPM"


def test_answer_question_returns_not_found_when_no_chunks_retrieved():
    with patch.object(mod, "hybrid_search", return_value=[]):
        result = mod.answer_question("What is the capital of France?")

    assert "couldn't find" in result.answer
    assert result.citations == []


def test_answer_question_returns_not_found_when_model_cites_nothing():
    chunk = _make_chunk()

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "complete", return_value="The excerpts don't contain this information.\nCITED:"
    ):
        result = mod.answer_question("What is the capital of France?")

    assert "couldn't find" in result.answer
    assert result.citations == []
