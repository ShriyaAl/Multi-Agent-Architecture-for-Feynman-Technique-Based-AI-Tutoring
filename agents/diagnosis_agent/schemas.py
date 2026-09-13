from pydantic import BaseModel, Field
from typing import List

class DimensionResult(BaseModel):
    score: int = Field(..., ge=0, le=2)  # 0 = fail, 1 = partial, 2 = fine
    matched_gap_ids: List[str] = Field(default_factory=list)
    reasoning: str  # short justification, for interpretability/logging

class DiagnosisResult(BaseModel):
    completeness: DimensionResult
    accuracy: DimensionResult
    clarity: DimensionResult
    coherence: DimensionResult

    def all_gap_tags(self) -> List[str]:
        tags = []
        for dim in [self.completeness, self.accuracy, self.clarity, self.coherence]:
            tags.extend(dim.matched_gap_ids)
        return tags