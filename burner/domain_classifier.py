import numpy as np
from burner.embedder import get_model
from burner.utils import cosine_similarity

def domain_classifier(statement: str) -> str:
    """
    Classifies a self-talk statement into one of the four canonical domains:
    'fitness', 'career', 'finance', 'social'.
    Uses keyword matching first, and falls back to semantic embedding similarity.
    """
    keywords = {
        "fitness": [
            "gym", "workout", "fitness", "run", "diet", "lift", "training", "exercise", 
            "cardio", "healthy", "nutrition", "athlete", "marathon", "protein", "gains", 
            "muscle", "sports", "stretch", "jog", "shape", "routine", "meals", "macros"
        ],
        "career": [
            "job", "career", "resume", "work", "boss", "promotion", "company", "sprint", 
            "office", "meeting", "professional", "project", "tasks", "code", 
            "programming", "refactor", "database", "api", "endpoints", "manager", "sprints"
        ],
        "finance": [
            "money", "finance", "invest", "stock", "budget", "spend", "savings", "debt", 
            "cash", "bank", "expense", "cost", "saving", "utility", "bill", "stocks"
        ],
        "social": [
            "social", "friend", "party", "meet", "talk", "family", "relationship", "date", 
            "hangout", "dinner", "call", "meetup", "colleague", "friends", "hang", "chat"
        ]
    }
    
    statement_lower = statement.lower()
    scores = {domain: 0 for domain in keywords}
    for domain, words in keywords.items():
        for word in words:
            # Match word boundary or prefix
            if word in statement_lower:
                scores[domain] += 1
                
    max_score = max(scores.values())
    winners = [d for d, s in scores.items() if s == max_score and s > 0]
    
    if len(winners) == 1:
        return winners[0]
        
    # Embedding fallback if keywords are ambiguous or absent
    model = get_model()
    
    domain_prototypes = {
        "fitness": ["health and fitness goals", "exercise and diet routine", "gym and sports workouts"],
        "career": ["professional career and work progress", "job tasks and office projects", "employment and business success"],
        "finance": ["money budgeting and investments", "financial savings and expenses", "stocks and bank accounts"],
        "social": ["social life and relationships", "family and hanging out with friends", "communication and meeting people"]
    }
    
    statement_vec = model.encode(statement, convert_to_numpy=True)
    best_domain = "fitness"
    best_sim = -1.0
    
    for domain, prototypes in domain_prototypes.items():
        proto_vecs = model.encode(prototypes, convert_to_numpy=True)
        sims = [cosine_similarity(statement_vec, pv) for pv in proto_vecs]
        mean_sim = float(np.mean(sims))
        if mean_sim > best_sim:
            best_sim = mean_sim
            best_domain = domain
            
    return best_domain
