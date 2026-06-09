import re
import numpy as np
from burner.output import DivergenceType
from burner.sufficiency import BehaviorEntry, parse_timestamp

def has_goal_oriented_language(talk: list[str]) -> bool:
    """Returns True if any self-talk string contains goal-oriented or future-intent verbs."""
    pattern = re.compile(
        r"\b(want to|going to|plan to|will|hope to|intend to|aim to|wish to|try to|should|must|have to|going\s+to|planning\s+to|would\s+like\s+to)\b",
        re.IGNORECASE
    )
    return any(pattern.search(t) is not None for t in talk)

def is_progress_declining(acts: list[BehaviorEntry]) -> bool:
    """
    Splits behavioral logs chronologically into two halves.
    Returns True if total behavioral intensity in the second half is <= first half.
    """
    if len(acts) <= 1:
        return True
        
    # Sort acts by timestamp
    sorted_acts = sorted(acts, key=lambda a: parse_timestamp(a.timestamp))
    dates = [parse_timestamp(a.timestamp) for a in sorted_acts]
    
    min_date = dates[0]
    max_date = dates[-1]
    if min_date == max_date:
        return True
        
    midpoint = min_date + (max_date - min_date) / 2
    
    first_half = [a for a, d in zip(sorted_acts, dates) if d <= midpoint]
    second_half = [a for a, d in zip(sorted_acts, dates) if d > midpoint]
    
    sum_first = sum(a.intensity for a in first_half)
    sum_second = sum(a.intensity for a in second_half)
    
    return sum_second <= sum_first

def is_blind_spot(
    domain: str,
    talk: list[str],
    acts: list[BehaviorEntry],
    all_acts: list[BehaviorEntry],
    min_talk_entries: int = 1
) -> bool:
    """
    Returns True if the domain has zero self-talk and its behavioral frequency
    is in the top quartile (upper 25%) of behavioral activity across all domains.
    """
    if len(talk) >= min_talk_entries:
        return False
        
    if not acts:
        return False
        
    # Count behavior frequencies for all domains
    counts = {}
    for a in all_acts:
        counts[a.domain] = counts.get(a.domain, 0) + 1
        
    if not counts:
        return False
        
    freqs = list(counts.values())
    threshold = float(np.percentile(freqs, 75))
    
    return len(acts) >= threshold

def classify_divergence(
    domain: str,
    score: float,
    talk: list[str],
    acts: list[BehaviorEntry],
    all_acts: list[BehaviorEntry],
    min_talk_entries: int = 1
) -> DivergenceType:
    """
    Determines the divergence classification category based on score and evidence patterns.
    """
    if is_blind_spot(domain, talk, acts, all_acts, min_talk_entries):
        return DivergenceType.BLIND_SPOT
        
    if score >= 0.35:
        if has_goal_oriented_language(talk) and is_progress_declining(acts):
            return DivergenceType.ASPIRATION_GAP
        return DivergenceType.OVERSTATEMENT
    elif score <= -0.35:
        return DivergenceType.UNDERSTATEMENT
    else:
        return DivergenceType.ALIGNED
