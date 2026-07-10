from unittest.mock import MagicMock, patch

from core.models import RetrievedChunk
import core.rag as mod


def _make_chunk(ticker="JPM", text="Net interest margin was 2.7%.", section="Item 7", score=0.02):
    return RetrievedChunk(
        ticker=ticker,
        company_name="JPMorgan Chase & Co.",
        filing_type="10-K",
        section=section,
        fiscal_year="2024",
        text=text,
        score=score,
    )


def _mock_anthropic_response(text: str):
    mock_content = MagicMock()
    mock_content.text = text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    return mock_response


def test_answer_question_returns_grounded_answer_with_citations():
    chunk = _make_chunk()
    fake_response = _mock_anthropic_response(
        "The net interest margin was 2.7% for the fiscal year.\nCITED: 0"
    )

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "get_anthropic_client"
    ) as mock_get_client:
        mock_get_client.return_value.messages.create.return_value = fake_response
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
    fake_response = _mock_anthropic_response(
        "The excerpts don't contain this information.\nCITED:"
    )

    with patch.object(mod, "hybrid_search", return_value=[chunk]), patch.object(
        mod, "get_anthropic_client"
    ) as mock_get_client:
        mock_get_client.return_value.messages.create.return_value = fake_response
        result = mod.answer_question("What is the capital of France?")

    assert "couldn't find" in result.answer
    assert result.citations == []
