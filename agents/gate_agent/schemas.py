from pydantic import BaseModel
from typing import Literal

class GateResult(BaseModel):
    decision: Literal["pass", "reject"]
    category: Literal["diagnosable", "no_attempt", "off_topic", "student_question"]
    reasoning: str