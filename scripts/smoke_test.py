from core.rag import answer_question

QUESTIONS = [
    "What was JPMorgan's net interest margin discussion in its most recent 10-K?",
    "How does Apple describe supply chain risk?",
    "What does Visa say about its net revenue?",
]


def main() -> None:
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        result = answer_question(question)
        print(f"A: {result.answer}")
        for citation in result.citations:
            print(f"  - {citation.company_name} / {citation.section} (FY{citation.fiscal_year})")


if __name__ == "__main__":
    main()
