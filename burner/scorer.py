import numpy as np
from datetime import datetime, timezone
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from burner.sufficiency import BehaviorEntry, SufficiencyConfig, check_sufficiency, parse_timestamp, most_recent
from burner.domain_classifier import domain_classifier
from burner.utils import cosine_similarity, analyze_sentiment, mean_embed

class DivergenceScore(BaseModel):
    domain: str
    score: float
    talk_count: int
    act_count: int

def activity_vector(domain_acts: list[BehaviorEntry], model: SentenceTransformer, ref_date: datetime) -> np.ndarray:
    """Computes the weighted centroid of behavior embeddings, adjusted by intensity and recency decay."""
    if not domain_acts:
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: 384)()
        return np.zeros(dim)
        
    texts = [a.text for a in domain_acts]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    weights = []
    for a in domain_acts:
        # Compute days ago relative to ref_date
        days_ago = most_recent([a], ref_date)
        recency_factor = np.exp(-0.05 * days_ago)
        weight = a.intensity * recency_factor
        weights.append(weight)
        
    weights = np.array(weights)
    if np.sum(weights) == 0:
        return np.mean(embeddings, axis=0)
    return np.average(embeddings, axis=0, weights=weights)

def narrative_density(domain_talk: list[str]) -> float:
    """Calculates the overall narrative mass using sentiment adjustment."""
    # Positive sentiment increases narrative mass, negative sentiment decreases it.
    # We use (1.0 + sentiment) to scale each talk snippet.
    return float(sum(1.0 + analyze_sentiment(s) for s in domain_talk))

def behavior_density(domain_acts: list[BehaviorEntry], ref_date: datetime) -> float:
    """Calculates the overall behavioral mass using intensity and recency decay."""
    mass = 0.0
    for a in domain_acts:
        days_ago = most_recent([a], ref_date)
        recency_factor = np.exp(-0.05 * days_ago)
        mass += a.intensity * recency_factor
    return float(mass)

def score_domain(
    self_talk: list[str],
    behavior_log: list[BehaviorEntry],
    domain: str,
    model: SentenceTransformer,
    cfg: SufficiencyConfig = None,
) -> DivergenceScore:
    """
    Returns a signed divergence score in [-1, +1] for a single domain.
    Positive  → narrative exceeds behavior (overstatement direction)
    Negative  → behavior exceeds narrative (understatement direction)
    0.0       → alignment
    """
    if cfg is None:
        cfg = SufficiencyConfig()
        
    ref_date = cfg.reference_date or datetime.now(timezone.utc)

    # 1. filter to domain
    domain_talk = [s for s in self_talk if domain_classifier(s) == domain]
    domain_acts  = [b for b in behavior_log if b.domain == domain]

    # 2. evidence sufficiency gate (raises InsufficientEvidence if sparse/stale)
    check_sufficiency(domain_talk, domain_acts, domain, cfg)

    # 3. embed and score
    # Use sentiment weight for mean embedding of self-talk
    talk_weights = [1.0 + analyze_sentiment(t) for t in domain_talk]
    talk_vec = mean_embed(domain_talk, model, weights=talk_weights)
    act_vec  = activity_vector(domain_acts, model, ref_date)
    raw_cos  = cosine_similarity(talk_vec, act_vec)

    # 4. sign by narrative vs behavior dominance
    narrative_mass = narrative_density(domain_talk)
    behavior_mass  = behavior_density(domain_acts, ref_date)
    sign = +1 if narrative_mass > behavior_mass else -1

    return DivergenceScore(
        domain=domain,
        score=float(sign * (1.0 - raw_cos)),
        talk_count=len(domain_talk),
        act_count=len(domain_acts),
    )
