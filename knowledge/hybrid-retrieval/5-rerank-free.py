import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

# We import the clean BM25 retriever and the corpus loader
from utils.retrievers import BM25Retriever, load_corpus

# Setup paths matching your directory structure
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "fiqa"
DENSE_DIR = ROOT / "indexes" / "dense"

# Initialize the free, open-source cross-encoder reranker model
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANK_MODEL)

K_RRF = 60

# --------------------------------------------------------------
# Step 1: Define local retrievers and fusion
# --------------------------------------------------------------

class LocalDenseRetriever:
    """A custom retriever that uses the free, local sentence-transformers model."""
    def __init__(self, corpus_df) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_ids = corpus_df["_id"].tolist()
        
        # Load the locally generated embedding matrix from the previous step
        matrix_path = DENSE_DIR / "embeddings_all-MiniLM-L6-v2.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing {matrix_path}. Run your embedding script first!")
            
        raw = np.load(matrix_path)
        self._embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        query_vec = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        query_vec /= np.linalg.norm(query_vec)
        scores = self._embeddings @ query_vec
        top_k = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k]


def local_hybrid_candidates(query: str, bm25: BM25Retriever, dense: LocalDenseRetriever, candidate_k: int = 50) -> list[tuple[str, float]]:
    """Fuses BM25 and Dense local rankings using Reciprocal Rank Fusion (RRF)."""
    bm25_ids = [doc_id for doc_id, _ in bm25.search(query, k=candidate_k)]
    dense_ids = [doc_id for doc_id, _ in dense.search(query, k=candidate_k)]
    
    scores: dict[str, float] = defaultdict(float)
    for ranking in [bm25_ids, dense_ids]:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (K_RRF + rank)
            
    return sorted(scores.items(), key=lambda x: -x[1])


# --------------------------------------------------------------
# Step 2: Load retrievers and corpus
# --------------------------------------------------------------

corpus_df = load_corpus()
corpus_by_id = corpus_df.set_index("_id")

bm25 = BM25Retriever()
dense = LocalDenseRetriever(corpus_df)


# --------------------------------------------------------------
# Step 3: Rerank with a local Cross-Encoder
# --------------------------------------------------------------


def search_reranked(
    query: str, k: int = 10, candidate_k: int = 50
) -> list[tuple[str, float]]:
    # Gather candidates locally using our open-source pipeline
    candidates = local_hybrid_candidates(query, bm25, dense, candidate_k=candidate_k)
    candidate_ids = [doc_id for doc_id, _ in candidates]
    candidate_texts = [corpus_by_id.loc[d, "text"] for d in candidate_ids]

    # Prepare inputs for the cross-encoder: pairs of [query, document_text]
    pairs = [[query, text] for text in candidate_texts]
    
    # Predict relevance scores (higher scores mean more relevant)
    relevance_scores = reranker.predict(pairs)

    # Combine IDs with scores and sort them in descending order
    reranked_results = list(zip(candidate_ids, relevance_scores))
    reranked_results.sort(key=lambda x: -x[1])

    return reranked_results[:k]


# --------------------------------------------------------------
# Step 4: Compare hybrid vs hybrid + rerank
# --------------------------------------------------------------


def show(label: str, results: list[tuple[str, float]]) -> None:
    print(f"\n{label}")
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = corpus_by_id.loc[doc_id, "text"]
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    show("Hybrid (RRF) only", local_hybrid_candidates(query, bm25, dense, candidate_k=50)[:5])
    show(f"Hybrid + Local Rerank ({RERANK_MODEL})", search_reranked(query, k=5))