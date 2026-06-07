import os
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize the open-source model (Runs completely free and locally)
# 'all-MiniLM-L6-v2' is fast and lightweight (384 dimensions vs OpenAI's 1536)
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

DATA_DIR = Path(__file__).parent / "data" / "fiqa"
INDEX_DIR = Path(__file__).parent / "indexes" / "dense"
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------
# Step 1: Embed the corpus in batches
# --------------------------------------------------------------


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts and return a numpy array."""
    # sentence-transformers handles the processing internally and returns a numpy array
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def build_index(doc_texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed the full corpus in batches with a progress bar."""
    chunks = []
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding"):
        chunks.append(embed_batch(doc_texts[i : i + batch_size]))
    return np.vstack(chunks)  # stack batches into one matrix


# --------------------------------------------------------------
# Step 2: Build or load the cached embedding matrix
# --------------------------------------------------------------

corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].tolist()

# Handle blank documents safely
doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]

# We include the model name in the filename to prevent loading an OpenAI cache matrix
embeddings_path = INDEX_DIR / f"embeddings_{model_name}.npy"

if embeddings_path.exists():
    print(f"Loading cached embeddings from {embeddings_path}")
    doc_embeddings = np.load(embeddings_path)
else:
    print(f"Embedding {len(doc_texts)} docs locally using {model_name} ($0.00 cost)")
    doc_embeddings = build_index(doc_texts)
    np.save(embeddings_path, doc_embeddings)

# Pre-normalize once so cosine similarity becomes a single dot product later.
doc_embeddings_normed = doc_embeddings / np.linalg.norm(
    doc_embeddings, axis=1, keepdims=True
)

# --------------------------------------------------------------
# Step 3: Query by cosine similarity
# --------------------------------------------------------------


def search_dense(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k (doc_id, similarity) pairs for a query."""
    query_vec = embed_batch([query])[0]
    query_vec /= np.linalg.norm(query_vec)
    scores = doc_embeddings_normed @ query_vec
    top_k = np.argsort(-scores)[:k]
    return [(doc_ids[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")
    for i, (doc_id, score) in enumerate(search_dense(query, k=5), 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}. [{score:.3f}] {doc_id} {text}\n")