from __future__ import annotations

from typing import Any

from prompts.classifier import build_ticket_classifier
from structured.schemas import TicketOutput

from .models import EvaluationResult


class TicketEvaluator:
    def __init__(self, generator):
        self.generator = generator

    def _resolve_ticket_text(self, row: Any) -> str:
        if hasattr(row, "to_dict"):
            data = row.to_dict()
        else:
            data = dict(row)

        for key in (
            "ticket",
            "text",
            "message",
            "description",
            "content",
            "query",
            "support_ticket",
            "customer_message",
        ):
            value = data.get(key)
            if value is not None:
                return str(value)

        for value in data.values():
            if isinstance(value, str) and value.strip():
                return value

        return ""

    def evaluate_ticket(self, row: Any) -> EvaluationResult:
        if hasattr(row, "to_dict"):
            data = row.to_dict()
        else:
            data = dict(row)

        ticket_id = int(data.get("ticket_id", 0))
        expected_category = str(data.get("category", ""))
        expected_sentiment = str(data.get("sentiment", ""))
        expected_urgency = str(data.get("urgency", ""))

        prompt = build_ticket_classifier(self._resolve_ticket_text(row))

        evaluation = EvaluationResult(
            ticket_id=ticket_id,
            success=False,
            expected_category=expected_category,
            expected_sentiment=expected_sentiment,
            expected_urgency=expected_urgency,
        )

        try:
            result = self.generator.generate(prompt, TicketOutput)
            evaluation.result = result
            evaluation.success = True

            if result is not None:
                evaluation.category_correct = getattr(result, "category", None) == expected_category
                evaluation.sentiment_correct = getattr(result, "sentiment", None) == expected_sentiment
                evaluation.urgency_correct = getattr(result, "urgency", None) == expected_urgency
        except Exception as exc:
            evaluation.error = str(exc)
            evaluation.success = False

        return evaluation

    def evaluate(self, dataframe) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []

        if hasattr(dataframe, "iterrows"):
            for _, row in dataframe.iterrows():
                results.append(self.evaluate_ticket(row))
            return results

        for row in dataframe:
            results.append(self.evaluate_ticket(row))

        return results
