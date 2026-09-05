"""
DisputeShield AI - RAG Index Builder (Milestone 3, real version)

Builds an actual retrieval index over the evidence documents in
synthetic_data/ (orders, shipping, communications, policies):

    documents -> chunk -> vectorize (TF-IDF) -> persist (joblib)

Then rag/retriever.py loads this index and does real cosine-similarity
top-K retrieval against it, filtered to a given order's documents plus
the global policy documents.

Why TF-IDF instead of neural embeddings (e.g. sentence-transformers):
neural embedding models need to be downloaded from a model hub the first
time they're used, which is an extra runtime dependency and a point of
failure that's hard to verify in every environment this project might run
in. TF-IDF is a real vector space model with genuine chunking, vectorizing,
and similarity search -- the actual RAG mechanics -- it's just a simpler
vectorizer. Swapping in a neural embedder later means changing exactly one
function (`_vectorize` below) without touching retriever.py's interface.

Run with:
    python rag/build_index.py
"""

import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).parent.parent.parent  # disputeshield/
SYNTHETIC_DATA = BASE_DIR / "synthetic_data"
INDEX_PATH = Path(__file__).parent / "index.joblib"


def _order_chunks(order_path: Path):
    """Turn one order.json into a human-readable text chunk, plus metadata.
    A "chunk" here is the whole small JSON record rendered as prose -- for
    documents this size, chunking further would just fragment useful
    context; the chunk boundary is naturally per-document."""
    order = json.loads(order_path.read_text())
    order_id = order["order_id"]
    text = (
        f"Order {order_id}: {order['product']}, amount {order['currency']} "
        f"{order['amount']}, ordered on {order['order_date']}, "
        f"current status: {order['status']}."
    )
    return [{"text": text, "order_id": order_id, "doc_type": "order",
              "source": str(order_path.relative_to(BASE_DIR))}]


def _tracking_chunks(tracking_path: Path):
    tracking = json.loads(tracking_path.read_text())
    order_id = tracking["order_id"]
    sig_text = "with a recipient signature on file" if tracking.get("delivered_signature") else "with no signature on file"
    text = (
        f"Shipment for order {order_id} via {tracking['carrier']}, "
        f"tracking number {tracking['tracking_number']}, "
        f"status: {tracking['status']}, {sig_text}, "
        f"delivery date {tracking.get('delivered_date', 'unknown')}."
    )
    return [{"text": text, "order_id": order_id, "doc_type": "tracking",
              "source": str(tracking_path.relative_to(BASE_DIR))}]


def _chat_chunks(chat_path: Path):
    order_id = chat_path.stem.replace("_chat", "")
    text = chat_path.read_text()
    return [{"text": text, "order_id": order_id, "doc_type": "chat",
              "source": str(chat_path.relative_to(BASE_DIR))}]


def _policy_chunks(policy_path: Path):
    """Global documents (not tied to one order_id) -- chunked per bullet
    point/paragraph, since these are the documents an LLM should actually
    be searching across meaningfully rather than reading whole."""
    text = policy_path.read_text()
    chunks = []
    for i, line in enumerate(l.strip() for l in text.split("\n") if l.strip().startswith("-")):
        chunks.append({
            "text": line.lstrip("- ").strip(),
            "order_id": None,  # global -- applies to every dispute
            "doc_type": "policy",
            "source": str(policy_path.relative_to(BASE_DIR)),
        })
    return chunks


def build_corpus():
    corpus = []
    for f in sorted((SYNTHETIC_DATA / "orders").glob("*.json")):
        corpus.extend(_order_chunks(f))
    for f in sorted((SYNTHETIC_DATA / "shipping").glob("*.json")):
        corpus.extend(_tracking_chunks(f))
    for f in sorted((SYNTHETIC_DATA / "communications").glob("*.txt")):
        corpus.extend(_chat_chunks(f))
    for f in sorted((SYNTHETIC_DATA / "policies").glob("*.md")):
        corpus.extend(_policy_chunks(f))
    return corpus


def main():
    corpus = build_corpus()
    texts = [c["text"] for c in corpus]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(texts)

    joblib.dump({
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadata": corpus,
    }, INDEX_PATH)

    print(f"Indexed {len(corpus)} chunks from synthetic_data/")
    by_type = {}
    for c in corpus:
        by_type[c["doc_type"]] = by_type.get(c["doc_type"], 0) + 1
    for k, v in by_type.items():
        print(f"  {k}: {v} chunks")
    print(f"Saved index -> {INDEX_PATH}")


if __name__ == "__main__":
    main()