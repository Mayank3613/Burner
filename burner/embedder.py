from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None

def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Lazy-loads and caches the SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model
