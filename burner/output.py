from enum import Enum
from pydantic import BaseModel, Field

class DivergenceType(str, Enum):
    OVERSTATEMENT = "OVERSTATEMENT"
    UNDERSTATEMENT = "UNDERSTATEMENT"
    BLIND_SPOT = "BLIND_SPOT"
    ASPIRATION_GAP = "ASPIRATION_GAP"
    ALIGNED = "ALIGNED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

class DomainResult(BaseModel):
    domain: str = Field(..., description="The life domain being analyzed")
    divergence_score: float | None = Field(default=None, description="Signed divergence score in [-1.0, +1.0]")
    divergence_type: DivergenceType = Field(..., description="Classification category for the gap")
    confidence: float | None = Field(default=None, description="Calculated confidence level of the classification in [0.0, 1.0]")
    evidence_count: int = Field(..., description="Total count of behavioral and self-talk data points analyzed")
    observation_days: int = Field(..., description="Number of calendar days spanned by the behavior data")
    abstained: bool = Field(..., description="True if evidence was insufficient to draw a classification")
    abstention_reason: str | None = Field(default=None, description="Reason for abstaining, if applicable")
