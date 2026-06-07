import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# We drop DenseRetriever from the import, but keep BM25 and the corpus loader
from utils.retrievers import BM25Retriever, load_corpus

K_RRF = 60
INDEX_DIR = Path(__file__).parent / "indexes" / "dense"


# --------------------------------------------------------------
# Step 1: The fusion function
# --------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = K_RRF
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of doc_ids into one ranked list."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# --------------------------------------------------------------
# Step 2: Define and Load both retrievers
# --------------------------------------------------------------

class LocalDenseRetriever:
    """A custom retriever that uses the free, local sentence-transformers model."""
    def __init__(self, corpus_df):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.doc_ids = corpus_df["_id"].tolist()
        
        # Load the locally generated embedding matrix from the previous step
        matrix_path = INDEX_DIR / "embeddings_all-MiniLM-L6-v2.npy"
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing {matrix_path}. Run the embedding script first!")
            
        embeddings = np.load(matrix_path)
        # Pre-normalize for fast cosine similarity dot products
        self.embeddings_normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        query_vec = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        query_vec /= np.linalg.norm(query_vec)
        scores = self.embeddings_normed @ query_vec
        top_k = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k]


# Load corpus and initialize both retrievers
corpus = load_corpus()
bm25 = BM25Retriever()
dense = LocalDenseRetriever(corpus)


# --------------------------------------------------------------
# Step 3: Search both, fuse, compare
# --------------------------------------------------------------


def search_hybrid(
    query: str, k: int = 10, candidate_k: int = 50
) -> list[tuple[str, float]]:
    """Retrieve top candidate_k from each retriever, fuse, return top k."""
    bm25_ids = [doc_id for doc_id, _ in bm25.search(query, k=candidate_k)]
    dense_ids = [doc_id for doc_id, _ in dense.search(query, k=candidate_k)]
    return reciprocal_rank_fusion([bm25_ids, dense_ids])[:k]


def show(label: str, results: list[tuple[str, float]]) -> None:
    print(f"\n{label}")
    for i, (doc_id, score) in enumerate(results[:5], 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"  {i}. [{score:.4f}] {doc_id}  {text[:70]}")


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"Query: {query}")

    show("BM25 only", bm25.search(query, k=5))
    show("Dense only", dense.search(query, k=5))
    show("Hybrid (RRF)", search_hybrid(query, k=5))