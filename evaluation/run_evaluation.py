import pandas as pd

from evaluation import (
    TicketEvaluator,
    category_accuracy,
    failure_count,
    sentiment_accuracy,
    urgency_accuracy,
    validity_rate,
)
from src.infrence.model import load_model
from structured.generator import StructuredGeneration


def main():
    tokenizer, model = load_model()
    generator = StructuredGeneration(model, tokenizer)
    tickets = pd.read_csv("data/tickets.csv")
    results = TicketEvaluator(generator).evaluate(tickets)

    print(f"validity_rate: {validity_rate(results):.2%}")
    print(f"failure_count: {failure_count(results)}")
    print(f"category_accuracy: {category_accuracy(results):.2%}")
    print(f"sentiment_accuracy: {sentiment_accuracy(results):.2%}")
    print(f"urgency_accuracy: {urgency_accuracy(results):.2%}")

    for result in results:
        if not result.success:
            print(f"ticket {result.ticket_id} failed: {result.error}")


if __name__ == "__main__":
    main()
