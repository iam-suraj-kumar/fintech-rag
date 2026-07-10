import streamlit as st

from core.models import RAGAnswer
from core.rag import answer_question

COMPANIES = [
    "Apple (AAPL)",
    "JPMorgan Chase (JPM)",
    "Visa (V)",
    "Bank of America (BAC)",
    "Wells Fargo (WFC)",
]


def render_answer(result: RAGAnswer) -> None:
    st.write(result.answer)
    if result.citations:
        with st.expander(f"Sources ({len(result.citations)})"):
            for citation in result.citations:
                st.markdown(
                    f"**{citation.company_name}** — {citation.section} "
                    f"(FY{citation.fiscal_year})"
                )
                st.caption(citation.text_snippet)


def main() -> None:
    st.set_page_config(page_title="FinTech RAG Demo")
    st.title("SEC Filing Q&A")

    with st.sidebar:
        st.subheader("Companies in this demo")
        for company in COMPANIES:
            st.write(f"- {company}")

    question = st.text_input("Ask a question about these filings")
    if st.button("Ask") and question:
        with st.spinner("Retrieving and generating answer..."):
            result = answer_question(question)
        render_answer(result)


if __name__ == "__main__":
    main()
