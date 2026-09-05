"""
DisputeShield AI - Agent State Machine (Milestone 5)

Wires together everything built so far into one runnable pipeline:

    DISPUTE_CREATED -> LOAD_DISPUTE -> GATHER_CONTEXT -> RETRIEVE_EVIDENCE
        -> CALCULATE_SCORE -> POLICY_GATE -> {SUBMIT | DRAFT | ESCALATE}
        -> AUDIT_LOG

Evidence retrieval here is STRUCTURED lookup (reading the JSON files
generate_dataset.py wrote to synthetic_data/) for the boolean flags used
by the ML model and policy gate -- those need to be exact deterministic
booleans, not similarity-search results.

The rebuttal-drafting layer is a REAL LLM call (rag/retriever.py finds
relevant evidence passages via TF-IDF vector search, llm_agent.py sends
them to Claude, and every claim the LLM returns is checked against the
retrieved evidence before being shown -- see llm_agent.verify_grounding()).
If no ANTHROPIC_API_KEY is set, this falls back to the deterministic
template generator further down in this file, so the whole pipeline still
runs end-to-end without an API key -- just without the LLM reasoning step.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from policy_engine import evaluate_policy, PolicyConfig, Decision
from llm_agent import generate_rebuttal_llm, llm_available

BASE_DIR = Path(__file__).parent.parent
SYNTHETIC_DATA = BASE_DIR / "synthetic_data"
DISPUTES_FILE = BASE_DIR / "ml" / "data" / "disputes.jsonl"
MODEL_FILE = BASE_DIR / "ml" / "models" / "contestability_model.joblib"

_rag_retriever = None  # lazy singleton -- loading the TF-IDF index is not
                        # free, so only load it once per process, and only
                        # if it's actually needed (LLM path available).


def get_rag_retriever():
    global _rag_retriever
    if _rag_retriever is None:
        from rag.retriever import RAGRetriever
        _rag_retriever = RAGRetriever()
    return _rag_retriever



class State(str, Enum):
    DISPUTE_CREATED = "DISPUTE_CREATED"
    LOAD_DISPUTE = "LOAD_DISPUTE"
    GATHER_CONTEXT = "GATHER_CONTEXT"
    RETRIEVE_EVIDENCE = "RETRIEVE_EVIDENCE"
    CALCULATE_SCORE = "CALCULATE_SCORE"
    POLICY_GATE = "POLICY_GATE"
    SUBMIT = "SUBMIT"
    DRAFT = "DRAFT"
    ESCALATE = "ESCALATE"
    AUDIT_LOG = "AUDIT_LOG"
    DONE = "DONE"


@dataclass
class AuditEntry:
    timestamp: str
    state: str
    detail: str


@dataclass
class AgentRun:
    dispute_id: str
    audit_trail: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    contestability_score: float = 0.0
    policy_result: dict = field(default_factory=dict)
    rebuttal: dict = field(default_factory=dict)
    rebuttal_source: str = "template"
    final_decision: str = ""
    action_result: dict = field(default_factory=dict)

    def log(self, state: State, detail: str):
        self.audit_trail.append(AuditEntry(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            state=state.value,
            detail=detail,
        ))

    def to_dict(self):
        return {
            "dispute_id": self.dispute_id,
            "final_decision": self.final_decision,
            "contestability_score": round(self.contestability_score, 3),
            "policy_result": self.policy_result,
            "rebuttal": self.rebuttal,
            "rebuttal_source": self.rebuttal_source,
            "action_result": self.action_result,
            "evidence_summary": {k: v for k, v in self.evidence.items()
                                  if k not in ("_raw_order", "_raw_tracking", "_raw_chat")},
            "audit_trail": [e.__dict__ for e in self.audit_trail],
        }


class EvidenceRetriever:
    """Structured lookup over the synthetic_data/ folder. Swap this class
    for an embeddings-backed retriever later; the interface stays the same."""

    def __init__(self, data_root: Path = SYNTHETIC_DATA):
        self.root = data_root

    def retrieve(self, order_id: str) -> dict:
        evidence = {
            "order_id": order_id,
            "delivery_confirmed": False,
            "tracking_available": False,
            "customer_acknowledged_delivery": False,
            "delivery_signature": False,
            "refund_issued": False,
            "sources": {},
        }

        order_path = self.root / "orders" / f"{order_id}.json"
        if order_path.exists():
            order = json.loads(order_path.read_text())
            evidence["delivery_confirmed"] = order.get("status") == "delivered"
            evidence["sources"]["order"] = str(order_path.relative_to(BASE_DIR))
            evidence["_raw_order"] = order

        tracking_path = self.root / "shipping" / f"{order_id}_tracking.json"
        if tracking_path.exists():
            tracking = json.loads(tracking_path.read_text())
            evidence["tracking_available"] = True
            evidence["delivery_signature"] = bool(tracking.get("delivered_signature"))
            evidence["sources"]["tracking"] = str(tracking_path.relative_to(BASE_DIR))
            evidence["_raw_tracking"] = tracking

        chat_path = self.root / "communications" / f"{order_id}_chat.txt"
        if chat_path.exists():
            evidence["customer_acknowledged_delivery"] = True
            evidence["sources"]["chat"] = str(chat_path.relative_to(BASE_DIR))
            evidence["_raw_chat"] = chat_path.read_text()

        return evidence


def load_model():
    """Loads the real persisted model (joblib) chosen by ml/train.py --
    either the isotonic-calibrated LogisticRegression or XGBoost, whichever
    won on held-out ROC-AUC. This replaces the old hand-rolled sigmoid/
    weights-in-json approach entirely, so there is exactly ONE place
    (sklearn's own predict_proba) that turns features into a score --
    no more risk of the training-time and serving-time math drifting
    apart (which is how the evidence_count/5-vs-4 bug happened before).
    """
    import joblib
    return joblib.load(MODEL_FILE)


def score_contestability(evidence: dict, model) -> float:
    """Builds the exact same 7-feature vector ml/train.py trains on,
    then calls the real model's predict_proba -- no reimplemented math."""
    checklist = ["delivery_confirmed", "tracking_available",
                 "customer_acknowledged_delivery", "delivery_signature"]
    evidence_count = sum(1 for f in checklist if evidence.get(f))

    x = [[
        1.0 if evidence.get("delivery_confirmed") else 0.0,
        1.0 if evidence.get("tracking_available") else 0.0,
        1.0 if evidence.get("customer_acknowledged_delivery") else 0.0,
        1.0 if evidence.get("delivery_signature") else 0.0,
        1.0 if evidence.get("refund_issued") else 0.0,
        evidence_count / 4.0,
        evidence.get("customer_previous_disputes_norm", 0.0),
    ]]
    return float(model.predict_proba(x)[0][1])


def generate_rebuttal(dispute: dict, evidence: dict, decision: Decision) -> dict:
    """
    Stubbed rebuttal generator (represents the LLM call in the full build).
    Every factual claim is paired with the evidence_id/source it came from,
    and the generator REFUSES to state a claim for which no evidence exists
    -- this is the "0 unsupported claims" contract from the plan.
    """
    claims = []

    def add_claim(text, source_key):
        source = evidence.get("sources", {}).get(source_key)
        if source:
            claims.append({"claim": text, "source": source})

    if evidence.get("delivery_confirmed"):
        add_claim("The order was fulfilled and marked delivered by the carrier.", "order")
    if evidence.get("tracking_available"):
        add_claim("Tracking records exist for this shipment.", "tracking")
    if evidence.get("delivery_signature"):
        add_claim("Delivery was confirmed with a recipient signature.", "tracking")
    if evidence.get("customer_acknowledged_delivery"):
        add_claim("The customer acknowledged receipt in a support conversation.", "chat")

    if decision == Decision.DO_NOT_CONTEST:
        body = ("Insufficient grounded evidence exists to support a rebuttal. "
                "No submission was drafted; escalate to manual investigation if needed.")
        claims = []
    else:
        lines = [f"- {c['claim']} (source: {c['source']})" for c in claims]
        body = (
            f"DISPUTE REPRESENTMENT\n\n"
            f"Dispute: {dispute['dispute_id']}  Amount: {dispute['amount']} {dispute.get('currency','INR')}\n"
            f"Reason: {dispute['reason_code']}\n\n"
            f"Supporting evidence:\n" + "\n".join(lines) + "\n\n"
            f"Based on the attached evidence, we request the dispute be rejected."
        )

    return {"body": body, "grounded_claims": claims, "unsupported_claim_count": 0}


def run_agent(dispute: dict, config: PolicyConfig = None, include_rebuttal: bool = False) -> AgentRun:
    run = AgentRun(dispute_id=dispute["dispute_id"])
    run.log(State.DISPUTE_CREATED, f"Dispute {dispute['dispute_id']} received, "
                                     f"amount={dispute['amount']} reason={dispute['reason_code']}")

    run.log(State.LOAD_DISPUTE, f"Loaded dispute record for payment {dispute['payment_id']}")

    run.log(State.GATHER_CONTEXT, f"Order={dispute['order_id']} customer={dispute['customer_id']}")

    retriever = EvidenceRetriever()
    evidence = retriever.retrieve(dispute["order_id"])
    evidence["refund_issued"] = dispute.get("refund_issued", False)
    evidence["customer_previous_disputes_norm"] = min(
        dispute.get("customer_previous_disputes", 0), 5) / 5.0
    run.evidence = evidence
    found = [k for k in ("delivery_confirmed", "tracking_available",
                          "customer_acknowledged_delivery") if evidence.get(k)]
    run.log(State.RETRIEVE_EVIDENCE, f"Retrieved evidence flags present: {found or 'none'}")

    model = load_model()
    score = score_contestability(evidence, model)
    run.contestability_score = score
    run.log(State.CALCULATE_SCORE, f"Contestability score = {score:.3f}")

    deadline_days = dispute.get("respond_by_days", 7)
    policy_result = evaluate_policy(
        contestability_score=score,
        evidence=evidence,
        amount=dispute["amount"],
        deadline_days_remaining=deadline_days,
        config=config,
    )
    run.policy_result = policy_result.to_dict()
    run.log(State.POLICY_GATE, f"Policy decision = {policy_result.decision.value}: "
                                 f"{policy_result.reasons[0]}")

    rebuttal = generate_rebuttal(dispute, evidence, policy_result.decision)
    rebuttal_source = "template"

    if include_rebuttal and policy_result.decision != Decision.DO_NOT_CONTEST and llm_available():
        try:
            retriever = get_rag_retriever()
            query = (f"Was the product delivered? Did the customer acknowledge "
                     f"receiving order {dispute['order_id']}? What does policy say "
                     f"about contesting a {dispute['reason_code']} dispute?")
            retrieved = retriever.retrieve(dispute["order_id"], query, top_k=6)
            llm_rebuttal = generate_rebuttal_llm(dispute, retrieved)
            rebuttal = llm_rebuttal
            rebuttal_source = f"llm ({llm_rebuttal.get('llm_model', 'unknown')})"
            run.log(State.AUDIT_LOG,
                    f"LLM rebuttal generated: {len(llm_rebuttal['grounded_claims'])} "
                    f"grounded claim(s), {llm_rebuttal['unsupported_claim_count']} "
                    f"claim(s) rejected by grounding check")
        except Exception as e:
            # Any LLM/API failure (rate limit, network, bad JSON, etc.)
            # falls back to the deterministic template rather than crashing
            # the whole pipeline -- a demo should degrade gracefully, not
            # go down because an external API call failed.
            run.log(State.AUDIT_LOG, f"LLM rebuttal generation failed ({e}); "
                                       f"falling back to deterministic template")

    run.rebuttal = rebuttal
    run.rebuttal_source = rebuttal_source
    if not include_rebuttal:
        run.log(State.AUDIT_LOG, "LLM rebuttal deferred until an explicit review/execution action")

    if policy_result.decision == Decision.AUTO_SUBMIT:
        run.log(State.SUBMIT, "Auto-submit authorized by policy gate; waiting for an explicit execution trigger")
        run.final_decision = Decision.AUTO_SUBMIT.value
    elif policy_result.decision == Decision.DRAFT_FOR_REVIEW:
        run.log(State.DRAFT, "Draft created and routed to human approval queue")
        run.final_decision = Decision.DRAFT_FOR_REVIEW.value
    elif policy_result.decision == Decision.ESCALATE:
        run.log(State.ESCALATE, "Escalated to human review / evidence investigation")
        run.final_decision = Decision.ESCALATE.value
    else:
        run.log(State.ESCALATE, "DO_NOT_CONTEST: hard policy conflict; no action authorized")
        run.final_decision = Decision.DO_NOT_CONTEST.value

    run.log(State.AUDIT_LOG, "Audit record finalized")
    return run


def execute_contest(dispute: dict, run: AgentRun):
    """Execute an already-authorized contest. This function is the only place
    where the agent pipeline performs an external Razorpay action. Listing or
    inspecting disputes never causes a side effect."""
    if run.final_decision not in (Decision.AUTO_SUBMIT.value, Decision.DRAFT_FOR_REVIEW.value):
        raise ValueError(f"Decision {run.final_decision} is not executable")

    from razorpay_integration.client import get_adapter
    from evidence_packet import build_packet

    packet = build_packet(dispute, run, BASE_DIR / "runtime" / "evidence_packets")
    adapter = get_adapter()
    doc_id = adapter.upload_document(str(packet), purpose="dispute_evidence")
    result = adapter.contest_dispute(
        dispute["dispute_id"], [doc_id],
        run.rebuttal.get("body", "Evidence attached."),
        action="submit", evidence_type="shipping_proof"
    )
    run.action_result = {
        "executed": True,
        "adapter": type(adapter).__name__,
        "document_id": doc_id,
        "response": result,
    }
    run.log(State.SUBMIT, f"Contest executed via {type(adapter).__name__}; document={doc_id}; response={result.get('status','unknown')}")
    return run


def load_disputes(n=None):
    rows = []
    with open(DISPUTES_FILE) as f:
        for line in f:
            rows.append(json.loads(line))
            if n and len(rows) >= n:
                break
    return rows


if __name__ == "__main__":
    disputes = load_disputes(n=5)
    for d in disputes:
        run = run_agent(d)
        print(f"\n{'='*70}\nDispute {d['dispute_id']} -> {run.final_decision}\n{'='*70}")
        for entry in run.audit_trail:
            print(f"  [{entry.timestamp}] {entry.state:20s} {entry.detail}")
        if run.rebuttal["body"]:
            print(f"\n--- Rebuttal ---\n{run.rebuttal['body']}")