import json
import os
import datetime
import math
import ollama

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "conversation_log.jsonl")
EMBED_MODEL = "nomic-embed-text"


def _get_embedding(text):
    """Turns text into a vector (list of numbers) capturing its meaning,
    using a small local embedding model via Ollama."""
    try:
        result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return result["embedding"]
    except Exception:
        # If Ollama/embedding model isn't available for any reason, fail
        # gracefully -- logging still works, just without search capability
        # for this entry.
        return None


def _cosine_similarity(vec_a, vec_b):
    """Measures how similar two vectors are, from -1 (opposite) to 1 (identical).
    This is the standard way to compare embeddings."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def log_exchange(user_input, reply):
    """Appends one exchange to the permanent log, along with its embedding
    for future semantic search. Never overwrites, never summarizes."""
    combined_text = f"{user_input} {reply}"
    embedding = _get_embedding(combined_text)

    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "user": user_input,
        "mimir": reply,
        "embedding": embedding
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def search_log(query, top_n=3, min_similarity=0.5):
    """Semantic search across the full log using embedding similarity.
    Finds past exchanges related in MEANING, not just shared exact words."""
    if not os.path.exists(LOG_FILE):
        return []

    query_embedding = _get_embedding(query)
    if query_embedding is None:
        return []

    scored = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_embedding = entry.get("embedding")
            if entry_embedding is None:
                continue

            similarity = _cosine_similarity(query_embedding, entry_embedding)
            if similarity >= min_similarity:
                scored.append((similarity, entry))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for score, entry in scored[:top_n]]


def format_matches_for_prompt(matches):
    """Turns search results into a clean text block for the system prompt."""
    if not matches:
        return ""
    lines = ["Here are potentially relevant past conversation excerpts (only reference these if they are genuinely useful to the current message -- do not force it):"]
    for m in matches:
        date_str = m["timestamp"].split("T")[0]
        lines.append(f"[{date_str}] You said: \"{m['user']}\" | Mimir replied: \"{m['mimir']}\"")
    return "\n".join(lines)