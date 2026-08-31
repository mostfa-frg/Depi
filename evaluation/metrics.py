from collections.abc import Sequence

from .models import EvaluationResult


def _successful_results(results: Sequence[EvaluationResult]) -> list[EvaluationResult]:
    return [result for result in results if result.success and result.result is not None]


def validity_rate(results: Sequence[EvaluationResult]) -> float:
    if not results:
        return 0.0
    return len(_successful_results(results)) / len(results)


def failure_count(results: Sequence[EvaluationResult]) -> int:
    return sum(1 for result in results if not result.success)


def category_accuracy(results: Sequence[EvaluationResult]) -> float:
    successful = _successful_results(results)
    if not successful:
        return 0.0
    return sum(1 for result in successful if result.category_correct) / len(successful)


def sentiment_accuracy(results: Sequence[EvaluationResult]) -> float:
    successful = _successful_results(results)
    if not successful:
        return 0.0
    return sum(1 for result in successful if result.sentiment_correct) / len(successful)


def urgency_accuracy(results: Sequence[EvaluationResult]) -> float:
    successful = _successful_results(results)
    if not successful:
        return 0.0
    return sum(1 for result in successful if result.urgency_correct) / len(successful)
