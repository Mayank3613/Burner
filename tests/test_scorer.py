import pytest
import numpy as np
from datetime import datetime, timezone
from burner.utils import cosine_similarity, analyze_sentiment, mean_embed
from burner.sufficiency import BehaviorEntry
from burner.scorer import activity_vector, narrative_density, behavior_density

def test_cosine_similarity():
    # Identical vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0
    
    # Orthogonal vectors
    v3 = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0
    
    # Zero vectors
    v_zero = np.array([0.0, 0.0, 0.0])
    assert cosine_similarity(v1, v_zero) == 0.0

def test_analyze_sentiment():
    # Positive words
    assert analyze_sentiment("I am having a good and excellent day.") > 0.0
    # Negative words
    assert analyze_sentiment("This was a bad and lazy effort.") < 0.0
    # Neutral/no match
    assert analyze_sentiment("I walked to the store.") == 0.0

def test_mean_embed():
    class MockModel:
        def encode(self, texts, convert_to_numpy=True):
            # Return simple vectors based on text content length
            return np.array([[len(t), 0.0] for t in texts])
            
        def get_sentence_embedding_dimension(self):
            return 2
            
    model = MockModel()
    
    # Empty texts
    assert np.array_equal(mean_embed([], model), np.zeros(2))
    
    # Single text
    assert np.array_equal(mean_embed(["hello"], model), np.array([5.0, 0.0]))
    
    # Multiple texts, equal weight
    assert np.array_equal(mean_embed(["abc", "abcde"], model), np.array([4.0, 0.0]))
    
    # Multiple texts, weighted
    # "abc" (length 3, weight 2.0), "abcde" (length 5, weight 0.0) -> centroid should be [3.0, 0.0]
    assert np.array_equal(
        mean_embed(["abc", "abcde"], model, weights=[2.0, 0.0]), 
        np.array([3.0, 0.0])
    )

def test_density_calculations():
    ref_date = datetime(2026, 6, 13, tzinfo=timezone.utc)
    
    talk = ["This is good", "This is bad"]
    # Positive sentiment of "good" makes weight 1.0 + 1.0 = 2.0
    # Negative sentiment of "bad" makes weight 1.0 - 1.0 = 0.0
    # Total narrative density should be 2.0
    assert pytest.approx(narrative_density(talk)) == 2.0
    
    # Behavior density
    acts = [
        # 1 day ago: recency factor = exp(-0.05 * 1) = 0.9512
        BehaviorEntry(timestamp="2026-06-12T00:00:00Z", text="Jog", domain="fitness", intensity=1.0),
        # 3 days ago: recency factor = exp(-0.05 * 3) = 0.8607, intensity 2.0 -> 1.7214
        BehaviorEntry(timestamp="2026-06-10T00:00:00Z", text="Gym", domain="fitness", intensity=2.0),
    ]
    expected_density = 1.0 * np.exp(-0.05 * 1) + 2.0 * np.exp(-0.05 * 3)
    assert pytest.approx(behavior_density(acts, ref_date)) == expected_density
