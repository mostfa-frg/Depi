import torch

from evaluation.metrics import (
    category_accuracy,
    failure_count,
    sentiment_accuracy,
    urgency_accuracy,
    validity_rate,
)
from evaluation.models import EvaluationResult
from prompts.classifier import build_ticket_classifier
from structured.parser import parse_and_validate_ticket
from structured.schemas import TicketOutput


def test_classifier_prompt_contains_ticket_text():
    prompt = build_ticket_classifier("The app crashes when I upload a file.")
    assert "The app crashes when I upload a file." in prompt
    assert "Classify the customer ticket." in prompt


def test_ticket_schema_accepts_valid_output():
    result = parse_and_validate_ticket(
        '{"category":"technical","sentiment":"negative",'
        '"urgency":"high","summary":"The app crashes during file upload."}'
    )
    assert isinstance(result, TicketOutput)
    assert result.category == "technical"


def test_ticket_schema_rejects_invalid_output():
    try:
        parse_and_validate_ticket('{"category":"unknown"}')
    except ValueError as exc:
        assert "MODEL output does not match ticket" in str(exc)
    else:
        raise AssertionError("Invalid ticket output should be rejected")


def test_evaluation_metrics():
    results = [
        EvaluationResult(
            ticket_id=1,
            success=True,
            expected_category="technical",
            expected_sentiment="negative",
            expected_urgency="high",
            result=TicketOutput(
                category="technical",
                sentiment="negative",
                urgency="high",
                summary="Broken laptop screen.",
            ),
            category_correct=True,
            sentiment_correct=True,
            urgency_correct=True,
        ),
        EvaluationResult(
            ticket_id=2,
            success=False,
            expected_category="account",
            expected_sentiment="neutral",
            expected_urgency="medium",
            error="generation failed",
        ),
    ]
    assert validity_rate(results) == 0.5
    assert failure_count(results) == 1
    assert category_accuracy(results) == 1.0
    assert sentiment_accuracy(results) == 1.0
    assert urgency_accuracy(results) == 1.0


def test_logit_bias():
    from src.infrence.decoder import apply_logit_bias

    logits = torch.zeros(1, 4)
    biased = apply_logit_bias(logits, {2: 3.0})
    assert biased[0, 2].item() == 3.0
    assert logits[0, 2].item() == 0.0
