from datetime import datetime, timezone
from pydantic import BaseModel, Field

class BehaviorEntry(BaseModel):
    timestamp: str
    text: str
    domain: str
    intensity: float = 1.0

class SufficiencyConfig(BaseModel):
    min_behavior_entries: int = Field(default=3, description="Minimum number of behavioral log entries required")
    min_talk_entries: int = Field(default=1, description="Minimum number of self-talk snippets required")
    min_span_days: int = Field(default=3, description="Minimum calendar day span required for behavioral evidence")
    stale_threshold_days: int = Field(default=14, description="Maximum age of the most recent behavioral log before data is stale")
    confidence_floor: float = Field(default=0.55, description="Confidence floor threshold for classification")
    reference_date: datetime | None = Field(default=None, description="Reference date for recency calculations (defaults to UTC now)")

class InsufficientEvidence(Exception):
    def __init__(self, reason: str, domain: str):
        self.reason = reason
        self.domain = domain
        super().__init__(f"[{domain}] Abstaining: {reason}")

def parse_timestamp(ts: str) -> datetime:
    """Safely parses timestamp strings including Z-suffixed ISO dates."""
    # Convert 'Z' to '+00:00' to be safe across different Python standard library versions
    cleaned = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)

def span_days(acts: list[BehaviorEntry]) -> int:
    """Calculates the unique number of calendar days spanned by the behavior entries."""
    if not acts:
        return 0
    dates = {parse_timestamp(a.timestamp).date() for a in acts}
    if not dates:
        return 0
    return (max(dates) - min(dates)).days + 1

def most_recent(acts: list[BehaviorEntry], ref_date: datetime) -> int:
    """Calculates the difference in days between ref_date and the most recent behavior entry."""
    if not acts:
        return 999999
    dates = [parse_timestamp(a.timestamp) for a in acts]
    max_date = max(dates)
    
    # Align timezone awareness
    if max_date.tzinfo is not None and ref_date.tzinfo is None:
        ref_date = ref_date.replace(tzinfo=timezone.utc)
    elif max_date.tzinfo is None and ref_date.tzinfo is not None:
        ref_date = ref_date.replace(tzinfo=None)
        
    delta = ref_date - max_date
    return delta.days

def check_sufficiency(
    talk: list[str],
    acts: list[BehaviorEntry],
    domain: str,
    cfg: SufficiencyConfig,
) -> None:
    """Asserts that the self-talk and behavior lists meet sufficiency criteria."""
    if len(acts) < cfg.min_behavior_entries:
        raise InsufficientEvidence("too few behavioral entries", domain)
        
    if len(talk) < cfg.min_talk_entries:
        raise InsufficientEvidence("no self-talk for domain", domain)
        
    if span_days(acts) < cfg.min_span_days:
        raise InsufficientEvidence("observation window too short", domain)
        
    ref_date = cfg.reference_date or datetime.now(timezone.utc)
    if most_recent(acts, ref_date) > cfg.stale_threshold_days:
        raise InsufficientEvidence("all data is stale", domain)
