"""
DisputeShield AI - RAG Retriever (Milestone 3, real version)

Loads the index built by build_index.py and performs REAL top-K retrieval:
given a dispute's order_id and a natural-language query, vectorizes the
query with the SAME fitted TF-IDF vectorizer, computes cosine similarity
against every chunk belonging to that order (plus the global policy
chunks, which apply to every dispute), and returns the top-K matches
ranked by similarity score.

This is deliberately NOT the same thing as agent.py's EvidenceRetriever
(structured key lookup). The two serve different jobs:

  - EvidenceRetriever (agent.py): produces exact boolean flags
    (delivery_confirmed=True/False) fed to the ML scoring model and the
    policy gate. Those need to be exact deterministic booleans -- a
    similarity score has no place deciding whether the policy gate's
    hard evidence-completeness check passes.

  - RAGRetriever (this file): produces ranked, relevant TEXT PASSAGES used
    to ground what the LLM reasons over and writes about in the rebuttal.
    This is where semantic retrieval actually adds value -- finding the
    passages that discuss delivery status, policy rules, etc., without
    the caller needing to already know the exact file path.

Keeping these separate is a deliberate design decision, not a shortcut --
mixing "similarity search" into a policy gate's hard evidence check would
make the gate non-deterministic, which defeats the entire point of having
a gate.
"""

from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path(__file__).parent / "index.joblib"


class RAGRetriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        index = joblib.load(index_path)
        self.vectorizer = index["vectorizer"]
        self.matrix = index["matrix"]
        self.metadata = index["metadata"]

        # Precompute which row indices belong to which order_id (and which
        # are global/policy chunks) so retrieval doesn't re-scan all ~18k
        # chunks' metadata on every call.
        self._order_index = {}
        self._global_rows = []
        for i, chunk in enumerate(self.metadata):
            if chunk["order_id"] is None:
                self._global_rows.append(i)
            else:
                self._order_index.setdefault(chunk["order_id"], []).append(i)

    def retrieve(self, order_id: str, query: str, top_k: int = 5):
        """Real retrieval: vectorize the query, compute cosine similarity
        against this order's chunks + global policy chunks, return top_k
        ranked by score (highest first)."""
        candidate_rows = self._order_index.get(order_id, []) + self._global_rows
        if not candidate_rows:
            return []

        query_vec = self.vectorizer.transform([query])
        candidate_matrix = self.matrix[candidate_rows]
        scores = cosine_similarity(query_vec, candidate_matrix)[0]

        ranked = sorted(zip(candidate_rows, scores), key=lambda x: -x[1])
        results = []
        for row_idx, score in ranked[:top_k]:
            chunk = self.metadata[row_idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "doc_type": chunk["doc_type"],
                "score": round(float(score), 4),
            })
        return results


if __name__ == "__main__":
    retriever = RAGRetriever()

    print("=== Query: 'was the package delivered and did the customer confirm it' ===")
    for r in retriever.retrieve("ORD_00002", "was the package delivered and did the customer confirm receiving it", top_k=4):
        print(f"  [{r['score']:.3f}] ({r['doc_type']}) {r['text'][:90]}  <- {r['source']}")

    print("\n=== Query: 'can we contest if a refund was already given' ===")
    for r in retriever.retrieve("ORD_00002", "can we contest a dispute if a refund was already issued", top_k=4):
        print(f"  [{r['score']:.3f}] ({r['doc_type']}) {r['text'][:90]}  <- {r['source']}")