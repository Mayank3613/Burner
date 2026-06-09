import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two numpy vectors. Returns 0.0 if either norm is zero."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def analyze_sentiment(text: str) -> float:
    """
    Returns a sentiment score in [-1.0, 1.0] based on a simple keyword lexicon.
    Positive words increase score, negative words decrease it.
    """
    pos_words = {
        "good", "great", "excellent", "consistent", "achieving", "active", "love", "like", "happy", "excited",
        "progress", "improving", "perfect", "strong", "better", "accomplished", "proud", "dedicated", "committed",
        "success", "succeed", "healthy", "fit", "clean", "productive", "efficient", "focused", "winning",
        "discipline", "disciplined", "achieve", "grow", "growth", "positive", "smart", "save", "saving", "invest"
    }
    neg_words = {
        "bad", "poor", "failing", "lazy", "inactive", "hate", "dislike", "sad", "frustrated", "struggling",
        "failed", "stale", "tired", "quit", "give up", "hard", "difficult", "worst", "unhealthy", "waste",
        "unproductive", "slow", "stressed", "neglect", "neglected", "lose", "losing", "wrong", "broke", "expensive"
    }
    
    words = text.lower().replace(".", "").replace(",", "").replace("!", "").split()
    if not words:
        return 0.0
        
    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)
    
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total

def mean_embed(texts: list[str], model, weights: list[float] = None) -> np.ndarray:
    """
    Computes the mean embedding vector of a list of strings using the provided SentenceTransformer.
    If weights are provided, computes the weighted average of the embeddings.
    """
    if not texts:
        # Return a zero vector matching the model's embedding dimension (MiniLM is 384)
        dim = getattr(model, "get_sentence_embedding_dimension", lambda: 384)()
        return np.zeros(dim)
        
    embeddings = model.encode(texts, convert_to_numpy=True)
    if len(texts) == 1:
        return embeddings[0]
        
    if weights is None:
        return np.mean(embeddings, axis=0)
        
    weights = np.array(weights)
    if np.sum(weights) == 0:
        return np.mean(embeddings, axis=0)
        
    # Ensure weights match dimensions for broadcasting
    return np.average(embeddings, axis=0, weights=weights)
