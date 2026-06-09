import pytest
from burner.output import DivergenceType
from burner.typer import classify_divergence
from burner.sufficiency import BehaviorEntry

# Mock helper to mimic the draft test signature
def sufficient():
    return "sufficient"

def run_burner(score: float, goal_language: bool = False, evidence: str = "sufficient") -> object:
    domain = "test_domain"
    
    if goal_language:
        talk = ["I want to run a marathon and plan to do it next year"]
    else:
        talk = ["I do standard activities and nothing else"]
        
    if evidence == "sufficient":
        # Setup behavior logs with declining progress over 3 days (midpoint split test)
        acts = [
            BehaviorEntry(timestamp="2026-06-01T08:00:00Z", text="Activity 1", domain=domain, intensity=1.0),
            BehaviorEntry(timestamp="2026-06-02T08:00:00Z", text="Activity 2", domain=domain, intensity=0.5),
            BehaviorEntry(timestamp="2026-06-03T08:00:00Z", text="Activity 3", domain=domain, intensity=0.2),
        ]
    else:
        acts = []
        
    div_type = classify_divergence(
        domain=domain,
        score=score,
        talk=talk,
        acts=acts,
        all_acts=acts
    )
    
    class MockResult:
        def __init__(self, dt):
            self.divergence_type = dt
            
    return MockResult(div_type)

def test_overstatement_at_threshold():
    """Score of exactly +0.35 should type as OVERSTATEMENT."""
    result = run_burner(score=+0.35, evidence=sufficient())
    assert result.divergence_type == DivergenceType.OVERSTATEMENT

def test_below_threshold_is_aligned():
    """Score of +0.34 should not type as overstatement."""
    result = run_burner(score=+0.34, evidence=sufficient())
    assert result.divergence_type == DivergenceType.ALIGNED

def test_aspiration_gap_requires_goal_language():
    """Aspiration gap must not trigger on overstatement-range score alone."""
    result = run_burner(score=+0.6, goal_language=False, evidence=sufficient())
    assert result.divergence_type != DivergenceType.ASPIRATION_GAP

def test_aspiration_gap_triggers_with_goal_language():
    """Aspiration gap triggers when score is positive, has goal language, and declining progress."""
    result = run_burner(score=+0.6, goal_language=True, evidence=sufficient())
    assert result.divergence_type == DivergenceType.ASPIRATION_GAP
