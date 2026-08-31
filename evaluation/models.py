from typing import Optional

from pydantic import BaseModel

from structured.schemas import TicketOutput


class EvaluationResult(BaseModel):
    ticket_id: int
    success: bool
    expected_category: str
    expected_sentiment: str
    expected_urgency: str
    result: Optional[TicketOutput] = None
    category_correct: bool = False
    sentiment_correct: bool = False
    urgency_correct: bool = False
    error: Optional[str] = None
