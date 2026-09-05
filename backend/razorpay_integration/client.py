"""
DisputeShield AI - Razorpay Integration (Milestone 7 cont.)

Implements the adapter/simulator pattern from the original plan:

                 DisputeShield
                      |
               RazorpayAdapter
                 /          \\
                /            \\
        RazorpayLiveClient   RazorpaySimulator

Both classes implement the same interface (fetch_dispute, upload_document,
contest_dispute). The rest of the system (agent.py, main.py) only ever talks
to `get_adapter()` -- it never knows or cares which implementation is behind
it. This means:

  - If RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET env vars are set, real test-mode
    API calls are made.
  - If they are NOT set (e.g. you haven't created a Razorpay test account
    yet, or you're demoing offline), the simulator kicks in automatically
    and the whole pipeline still runs end-to-end using the synthetic
    dataset -- nothing in agent.py or main.py has to change.

Field names, endpoints and the contest payload shape below are taken from
Razorpay's current public API docs (razorpay.com/docs/api/disputes/), not
memory -- see the docstrings on each method for the source.
"""

import hashlib
import hmac
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    import razorpay
except ImportError:
    razorpay = None


class RazorpayAdapterBase(ABC):
    @abstractmethod
    def fetch_dispute(self, dispute_id: str) -> dict:
        ...

    @abstractmethod
    def upload_document(self, file_path: str, purpose: str = "dispute_evidence") -> str:
        """Returns a document_id."""
        ...

    @abstractmethod
    def contest_dispute(self, dispute_id: str, document_ids: list, summary: str,
                         action: str = "draft", evidence_type: str = "shipping_proof") -> dict:
        """action is 'draft' or 'submit' per Razorpay's contest API --
        drafts are NOT auto-submitted, which is exactly the distinction
        our policy gate relies on (agent recommends CONTEST, policy gate
        decides whether that becomes a real submit)."""
        ...


class RazorpayLiveClient(RazorpayAdapterBase):
    """
    Thin wrapper over the official `razorpay` Python SDK, restricted to
    Test Mode keys. Endpoints used (per razorpay.com/docs/api/disputes/):

      GET  /v1/disputes/:id                  -> fetch_dispute
      POST /v1/documents                     -> upload_document
      PATCH /v1/disputes/:id/contest         -> contest_dispute

    The contest payload uses billing_proof / shipping_proof / others
    (list of {type, document_ids}) plus amount, summary, and action
    ("draft" leaves it as a saved draft; "submit" confirms the contest --
    Razorpay does NOT auto-submit a draft, so this distinction is load-
    bearing for our policy gate, not just an implementation detail).
    """

    def __init__(self, key_id: str, key_secret: str):
        if razorpay is None:
            raise RuntimeError("razorpay package not installed. pip install razorpay")
        self.client = razorpay.Client(auth=(key_id, key_secret))

    def fetch_dispute(self, dispute_id: str) -> dict:
        return self.client.dispute.fetch(dispute_id)

    def upload_document(self, file_path: str, purpose: str = "dispute_evidence") -> str:
        with open(file_path, "rb") as f:
            result = self.client.document.create({
                "file": f,
                "purpose": purpose,
            })
        return result["id"]

    def contest_dispute(self, dispute_id: str, document_ids: list, summary: str,
                         action: str = "draft", evidence_type: str = "shipping_proof") -> dict:
        allowed = {
            "shipping_proof", "billing_proof", "cancellation_proof",
            "customer_communication", "proof_of_service", "explanation_letter",
            "refund_confirmation", "access_activity_log",
            "refund_cancellation_policy", "term_and_conditions", "others",
        }
        if evidence_type not in allowed:
            raise ValueError(f"Unsupported Razorpay evidence type: {evidence_type}")
        payload = {
            evidence_type: document_ids,
            "summary": summary[:1000],
            "action": action,
        }
        return self.client.dispute.contest(dispute_id, payload)

    @staticmethod
    def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> bool:
        """
        Per Razorpay docs: HMAC-SHA256 (hex) over the RAW request body,
        keyed with the webhook secret (distinct from the API key/secret).
        Must use the raw bytes, not a re-serialized/parsed version of the
        JSON -- re-serializing can produce a different byte sequence and
        silently break verification.
        """
        expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)


class RazorpaySimulator(RazorpayAdapterBase):
    """
    Local fallback used when no Razorpay test credentials are configured,
    or when we deliberately don't want to hit the live API during a demo
    (flaky network, rate limits, etc). Reads/writes against the same
    synthetic_data/ folder the rest of the system already uses, so the
    full pipeline is demonstrable offline with zero external dependency.

    IMPORTANT: this is explicitly NOT pretending to be Razorpay. Every
    response includes "simulated": true so nothing downstream can
    mistake a local mock result for a real API submission.
    """

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = data_root or (Path(__file__).parent.parent.parent / "synthetic_data")
        self._uploaded_docs = {}

    def fetch_dispute(self, dispute_id: str) -> dict:
        return {
            "id": dispute_id,
            "entity": "dispute",
            "status": "under_review",
            "simulated": True,
        }

    def upload_document(self, file_path: str, purpose: str = "dispute_evidence") -> str:
        digest = hashlib.sha256(str(Path(file_path).resolve()).encode()).hexdigest()[:12]
        doc_id = f"doc_sim_{digest}"
        self._uploaded_docs[doc_id] = {"file_path": file_path, "purpose": purpose}
        return doc_id

    def contest_dispute(self, dispute_id: str, document_ids: list, summary: str,
                         action: str = "draft", evidence_type: str = "shipping_proof") -> dict:
        return {
            "id": dispute_id,
            "status": "under_review" if action == "draft" else "contest_submitted",
            "action_taken": action,
            "document_ids": document_ids,
            "evidence_type": evidence_type,
            "summary": summary,
            "simulated": True,
        }


def get_adapter() -> RazorpayAdapterBase:
    """
    Factory used by the rest of the app. Picks the live client if test-mode
    credentials are present in the environment, otherwise falls back to the
    simulator. This is the ONE place that decision is made -- agent.py and
    main.py just call get_adapter() and don't need to know which they got.
    """
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    key_secret = os.environ.get("RAZORPAY_KEY_SECRET")
    if key_id and key_secret:
        if key_id.startswith("rzp_live_"):
            raise RuntimeError(
                "Refusing to start with a LIVE Razorpay key in a buildathon demo project. "
                "Use a rzp_test_ key from Dashboard > Test Mode."
            )
        return RazorpayLiveClient(key_id, key_secret)
    return RazorpaySimulator()


if __name__ == "__main__":
    # Smoke test -- runs the simulator path since no real keys are set here.
    adapter = get_adapter()
    print(f"Using adapter: {type(adapter).__name__}")
    doc_id = adapter.upload_document("synthetic_data/orders/ORD_00003.json")
    print(f"Uploaded doc -> {doc_id}")
    result = adapter.contest_dispute("DSP_00003", [doc_id], "Evidence attached.", action="submit")
    print(result)