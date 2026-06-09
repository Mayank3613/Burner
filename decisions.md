# Design Decisions & Calibration Rationale — Burner

This document details the architectural choices, mathematical formulations, threshold parameters, and structural constraints implemented in the Burner divergence scoring engine.

---

## 1. Alignment Method

### Choice: sentence-transformers (`all-MiniLM-L6-v2`) with Signed Cosine Gap
We embed both self-talk snippets and behavior descriptions into a dense vector space using the `all-MiniLM-L6-v2` sentence-transformer model. The raw cosine similarity ($cos(\vec{v}_{talk}, \vec{v}_{act})$) is calculated between the sentiment-adjusted self-talk centroid and the activity-weighted behavior vector.

The final divergence score is defined as:
$$S = \text{sign} \times (1.0 - \text{cosine\_similarity})$$

Where $\text{sign}$ is determined by the relative density/mass of the self-talk narrative versus behavioral logs:
- $\text{sign} = +1$ if Narrative Mass > Behavior Mass (Overstatement direction)
- $\text{sign} = -1$ if Narrative Mass < Behavior Mass (Understatement direction)

### Rationale for the Chosen Method
- **Fast and Local**: Runs completely offline, requiring no external LLM API calls, credentials, or network requests after the initial model cache.
- **Semantic Mapping**: Avoids lexical mismatch issues by embedding different vocabularies (e.g., "I will run tomorrow" vs. "Completed 5km jog") into a shared semantic space.
- **Reproducible and Deterministic**: Unlike generative models, the embedding pipeline is deterministic and auditable.

### Rejected Alternatives
- **TF-IDF Keyword Vectorization**: Rejected because it fails to map synonyms or semantic relationships (e.g., mapping "gym" to "workout").
- **LLM/GPT-based Scoring**: Rejected because it introduces non-deterministic outputs, latency, runtime costs, and dependencies on external network APIs, violating local execution requirements.

---

## 2. Type Boundaries & Decision Rules

Every domain that meets sufficiency criteria is classified into exactly one of five types based on the following explicit rules:

| Type | Name | Criteria | Rationale |
| :--- | :--- | :--- | :--- |
| **Type 01** | `OVERSTATEMENT` | Score $> +0.35$ AND does not meet `ASPIRATION_GAP` conditions. | Self-talk is present and highly positive, but behavioral density is low or mismatched. |
| **Type 02** | `UNDERSTATEMENT` | Score $< -0.35$ AND does not meet `BLIND_SPOT` conditions. | Behavioral logging is heavy and consistent, but the user rarely mentions the domain in self-talk. |
| **Type 03** | `BLIND_SPOT` | Domain is in the top quartile (upper 25%) of behavioral frequency across all domains, and has zero self-talk entries. | Surfaces areas of significant active behavior that are completely absent from the user's self-concept narrative. Checked before scoring. |
| **Type 04** | `ASPIRATION_GAP` | Score $> +0.35$ AND self-talk contains goal-oriented language AND behavioral progress is flat/declining. | Distinguishes constructive but unfulfilled goal-setting from general overstatement. |
| **Type 05** | `ALIGNED` | Score is within $[-0.35, +0.35]$ | Stated narrative aligns closely with logged activity patterns. |

### Configuration Thresholds
- **Divergence Cutoff**: $\pm 0.35$. Scores exceeding this magnitude represent a meaningful semantic/volume disconnect.
- **Goal Language Regex**: Matches verbs indicating intention or obligation (`want to`, `going to`, `plan to`, `will`, `hope to`, `intend to`, `should`, `must`, `have to`).
- **Decline Detection**: The behavior log is split chronologically into two halves. If the total behavioral intensity in the second half is less than or equal to the first half, progress is flagged as flat/declining.

---

## 3. Evidence Sufficiency & Abstention Rules

Premature classification is treated as a core failure mode. The following gates are enforced before typing is attempted:

1. **Minimum Behavioral Entries (`min_behavior_entries: 3`)**: At least 3 behavior logs must exist.
2. **Minimum Self-talk Snippets (`min_talk_entries: 1`)**: At least 1 self-talk entry must exist (except in pre-checked `BLIND_SPOT` cases).
3. **Minimum Temporal Span (`min_span_days: 3`)**: Behavioral entries must span at least 3 distinct calendar days.
4. **Recency Window (`stale_threshold_days: 14`)**: If the most recent behavioral log is older than 14 days relative to the reference date, all data is considered stale.
5. **Confidence Floor (`confidence_floor: 0.55`)**: If the computed confidence score is below 0.55, the result is marked as abstained/uncertain.

*Failure of any sufficiency check results in an immediate abstention with `abstained = true` and a specific `abstention_reason` returned in `DomainResult`.*

---

## 4. Output Safety: Prohibited Claims

The output schema is designed to make characterological, clinical, or diagnostic claims **physically impossible**.
- **No Character Adjectives**: Fields like "lazy", "honest", "deceptive", "disciplined", "deluded", "avoidant", or "motivated" are completely omitted from the schema.
- **No Clinical Inference**: No mention of traits, disorders, personality types, or mental state diagnoses.
- **Gap-Only Scope**: The engine describes only the measured relationship between two signals: self-talk and behavior log text. It remains completely silent on the *why* or the *who*.

---

## 5. Known Limitations & Failure Modes

1. **Domain Jargon**: Sentence embeddings may struggle to align highly specialized terminology or abbreviations if they lie outside the standard vocabularies of the model.
2. **Volume-Insensitivity of Cosine Similarity**: Cosine similarity measures angle, not magnitude. A person saying "I love running" once vs. twenty times produces similar centroids. Burner mitigates this by weighting the sign and scores using density differences.
3. **Temporal Sparsity**: Short observation windows cannot detect long-term shifts in behavioral patterns.
4. **Language Barriers**: The default model is trained primarily on English data. Applying it to other languages without switching the model to a multilingual variant (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) will result in low similarity and incorrect classifications.
