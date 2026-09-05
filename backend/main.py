"""
DisputeShield AI - API Layer

FastAPI service for the DisputeShield AI chargeback-defense agent.

The dashboard uses a precomputed snapshot for the bundled 10,000-row
synthetic dataset so list/stat requests stay fast. Individual dispute
detail pages still run the full ML/RAG/LLM agent pipeline.

Human approvals are kept in memory for this hackathon MVP.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from pathlib import Path
import os
import json
import hashlib

from dotenv import load_dotenv

load_dotenv()

from backend.agent import (
    run_agent,
    load_disputes,
    load_model,
    execute_contest,
)
from backend.policy_engine import Decision
from backend.razorpay_integration.client import (
    get_adapter,
    RazorpayLiveClient,
)


app = FastAPI(
    title="DisputeShield AI",
    description="Autonomous chargeback defense agent -- Razorpay AI Buildathon",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo only
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_ALL_DISPUTES = {d["dispute_id"]: d for d in load_disputes()}

_RUN_CACHE = {}          # dispute_id -> AgentRun.to_dict()
_HUMAN_DECISIONS = {}    # dispute_id -> approval/rejection record

# Webhook delivery is at-least-once (Razorpay's own docs say the same
# event can arrive more than once), so this is a real production hazard,
# not a hypothetical -- without it, a retried webhook for an AUTO_SUBMIT
# dispute would call execute_contest() a second time. Keyed on Razorpay's
# x-razorpay-event-id header (the field Razorpay's own webhook docs name
# as "unique per event" and recommend for exactly this deduplication),
# falling back to a hash of the raw body if that header is ever missing
# (e.g. a hand-crafted test payload) so we still fail safe rather than
# skipping the dedup check entirely.
_PROCESSED_WEBHOOK_EVENTS = {}   # event_id -> {"received_at": iso str, "dispute_id": str, "action": str}

_razorpay_adapter = get_adapter()


# ---------------------------------------------------------------------------
# Fast dashboard snapshot
# ---------------------------------------------------------------------------

SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "ml"
    / "models"
    / "dashboard_snapshot.json"
)

if SNAPSHOT_PATH.exists():
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        _DASHBOARD_SNAPSHOT = json.load(f)
else:
    _DASHBOARD_SNAPSHOT = None

_DASHBOARD_ITEMS = {}

if _DASHBOARD_SNAPSHOT:
    _DASHBOARD_ITEMS = {
        item["dispute_id"]: item
        for item in _DASHBOARD_SNAPSHOT.get("items", [])
    }


class ApprovalRequest(BaseModel):
    action: str  # "approve" or "reject"
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Basic endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "DisputeShield AI",
        "status": "running",
        "disputes_loaded": len(_ALL_DISPUTES),
        "docs": "/docs",
        "razorpay_mode": type(_razorpay_adapter).__name__,
    }


@app.get("/health")
def health():
    model = load_model()

    return {
        "status": "ok",
        "model_type": type(model).__name__,
        "model_calibrated": True,
        "razorpay_mode": type(_razorpay_adapter).__name__,
    }


# ---------------------------------------------------------------------------
# Razorpay webhook
# ---------------------------------------------------------------------------

@app.post("/webhooks/razorpay")
async def razorpay_webhook(request: Request):
    """
    Receive Razorpay dispute webhooks.

    Signature verification uses the raw request body, matching Razorpay's
    webhook verification requirement.

    In simulator mode, when no webhook secret is configured, verification
    is skipped so the local demo remains usable without pretending that a
    real signed webhook was received.
    """

    raw_body = await request.body()

    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    signature = request.headers.get("x-razorpay-signature", "")

    verified = None

    if secret:
        verified = RazorpayLiveClient.verify_webhook_signature(
            raw_body,
            signature,
            secret,
        )

        if not verified:
            raise HTTPException(
                status_code=400,
                detail="Webhook signature verification failed",
            )

    payload = await request.json()
    event = payload.get("event", "unknown")

    # Idempotency: Razorpay webhook delivery is at-least-once, so the same
    # event can legitimately arrive more than once (retries, redelivery
    # after a timeout, etc). x-razorpay-event-id is unique per event --
    # that's the field Razorpay's own docs recommend for this exact check.
    # Falling back to a raw-body hash only if that header is ever absent,
    # so a malformed/test request still gets a stable dedup key instead of
    # silently skipping the check.
    event_id = request.headers.get("x-razorpay-event-id") or (
        "body-hash:" + hashlib.sha256(raw_body).hexdigest()
    )

    if event_id in _PROCESSED_WEBHOOK_EVENTS:
        return {
            "received": True,
            "event": event,
            "signature_verified": verified,
            "event_id": event_id,
            "status": "duplicate_ignored",
            "first_processed": _PROCESSED_WEBHOOK_EVENTS[event_id],
        }

    dispute_id = (
        payload.get("dispute_id")
        or payload.get("payload", {})
        .get("dispute", {})
        .get("entity", {})
        .get("id")
    )

    if not dispute_id or dispute_id not in _ALL_DISPUTES:
        _PROCESSED_WEBHOOK_EVENTS[event_id] = {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "dispute_id": dispute_id,
            "action": "ignored",
        }
        return {
            "received": True,
            "event": event,
            "signature_verified": verified,
            "action": "ignored -- unknown or missing dispute_id for this demo dataset",
        }

    # Record the event as processed BEFORE doing the (synchronous,
    # non-yielding) execution work below, so a retry that arrives while
    # this same request is still running is still recognized as a
    # duplicate rather than racing it.
    _PROCESSED_WEBHOOK_EVENTS[event_id] = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "dispute_id": dispute_id,
        "action": "processing",
    }

    run = _get_or_run(dispute_id)
    execution = None

    if (
        run["final_decision"] == Decision.AUTO_SUBMIT.value
        and not run.get("action_result", {}).get("executed")
    ):
        try:
            agent_run = run_agent(
                _ALL_DISPUTES[dispute_id],
                include_rebuttal=True,
            )

            executed = execute_contest(
                _ALL_DISPUTES[dispute_id],
                agent_run,
            )

            executed_dict = executed.to_dict()

            executed_dict = _add_execution_audit(
                executed_dict,
                executed=True,
            )

            _RUN_CACHE[dispute_id] = executed_dict
            execution = executed.action_result
            _PROCESSED_WEBHOOK_EVENTS[event_id]["action"] = "executed"

        except Exception as exc:
            _PROCESSED_WEBHOOK_EVENTS[event_id]["action"] = "failed"
            raise HTTPException(
                status_code=502,
                detail=f"Auto-submit execution failed: {exc}",
            )
    else:
        _PROCESSED_WEBHOOK_EVENTS[event_id]["action"] = "no_action_needed"

    return {
        "received": True,
        "event": event,
        "signature_verified": verified,
        "event_id": event_id,
        "dispute_id": dispute_id,
        "agent_decision": run["final_decision"],
        "execution": execution,
    }


# ---------------------------------------------------------------------------
# Dispute list
# ---------------------------------------------------------------------------

@app.get("/disputes")
def list_disputes(
    limit: int = 50,
    decision: Optional[str] = None,
):
    """
    Fast, read-only dispute list.

    Uses the precomputed ML/policy snapshot rather than running the full
    agent for every row. Detailed analysis is performed only when a
    merchant opens an individual dispute.
    """

    limit = max(1, min(limit, 500))

    results = []

    if _DASHBOARD_SNAPSHOT and _DASHBOARD_ITEMS:
        for item in _DASHBOARD_SNAPSHOT.get("items", []):
            if decision and item["decision"] != decision.upper():
                continue

            dispute_id = item["dispute_id"]
            dispute = _ALL_DISPUTES.get(dispute_id)

            if not dispute:
                continue

            results.append(
                {
                    "dispute_id": dispute_id,
                    "amount": dispute["amount"],
                    "reason_code": dispute["reason_code"],
                    "respond_by_days": dispute["respond_by_days"],
                    "contestability_score": item["contestability_score"],
                    "decision": item["decision"],
                    "human_decision": _HUMAN_DECISIONS.get(dispute_id),
                }
            )

            if len(results) >= limit:
                break

        return {
            "count": len(results),
            "disputes": results,
        }

    # Safe development fallback if the snapshot is missing.
    for dispute_id, dispute in list(_ALL_DISPUTES.items())[:limit]:
        run = _get_or_run(dispute_id)

        if decision and run["final_decision"] != decision.upper():
            continue

        results.append(
            {
                "dispute_id": dispute_id,
                "amount": dispute["amount"],
                "reason_code": dispute["reason_code"],
                "respond_by_days": dispute["respond_by_days"],
                "contestability_score": run["contestability_score"],
                "decision": run["final_decision"],
                "human_decision": _HUMAN_DECISIONS.get(dispute_id),
            }
        )

    return {
        "count": len(results),
        "disputes": results,
    }


# ---------------------------------------------------------------------------
# Individual dispute detail
# ---------------------------------------------------------------------------

@app.get("/disputes/{dispute_id}")
def get_dispute_detail(dispute_id: str):
    """
    Full detail for one dispute:
    evidence, model score, policy reasons, grounded rebuttal and audit trail.
    """

    if dispute_id not in _ALL_DISPUTES:
        raise HTTPException(
            status_code=404,
            detail=f"Dispute {dispute_id} not found",
        )

    # Explicit user action: run the complete analysis pipeline.
    base = run_agent(
        _ALL_DISPUTES[dispute_id],
        include_rebuttal=True,
    ).to_dict()

    base["human_decision"] = _HUMAN_DECISIONS.get(dispute_id)

    # If an execution happened earlier in this session, preserve the
    # execution result/audit trail instead of hiding it behind a fresh run.
    cached = _RUN_CACHE.get(dispute_id)

    if cached and cached.get("action_result", {}).get("executed"):
        base["action_result"] = cached["action_result"]
        base["audit_trail"] = cached.get("audit_trail", base.get("audit_trail", []))

    _RUN_CACHE[dispute_id] = base

    return base


# ---------------------------------------------------------------------------
# Human approval / rejection
# ---------------------------------------------------------------------------

@app.post("/disputes/{dispute_id}/approve")
def approve_dispute(
    dispute_id: str,
    req: ApprovalRequest,
):
    """
    Human approval is a real execution boundary.

    A merchant approval can execute a draft defense.
    A hard DO_NOT_CONTEST policy block cannot be overridden here.
    """

    if dispute_id not in _ALL_DISPUTES:
        raise HTTPException(
            status_code=404,
            detail=f"Dispute {dispute_id} not found",
        )

    if req.action not in ("approve", "reject"):
        raise HTTPException(
            status_code=400,
            detail="action must be 'approve' or 'reject'",
        )

    run = _get_or_run(dispute_id)

    if run["final_decision"] == Decision.DO_NOT_CONTEST.value:
        raise HTTPException(
            status_code=409,
            detail="Hard policy block cannot be overridden here.",
        )

    # ---------------------------------------------------------------
    # Human rejection
    # ---------------------------------------------------------------

    if req.action == "reject":
        rejected_run = dict(run)

        rejected_run = _add_execution_audit(
            rejected_run,
            rejected=True,
        )

        _RUN_CACHE[dispute_id] = rejected_run

        _HUMAN_DECISIONS[dispute_id] = {
            "action": "rejected",
            "note": req.note,
        }

        return {
            "dispute_id": dispute_id,
            "recorded": _HUMAN_DECISIONS[dispute_id],
        }

    # ---------------------------------------------------------------
    # Human approval
    # ---------------------------------------------------------------

    agent_run = run_agent(
        _ALL_DISPUTES[dispute_id],
        include_rebuttal=True,
    )

    if agent_run.final_decision not in (
        Decision.DRAFT_FOR_REVIEW.value,
        Decision.ESCALATE.value,
        Decision.AUTO_SUBMIT.value,
    ):
        raise HTTPException(
            status_code=409,
            detail="This dispute is not eligible for human-approved execution.",
        )

    if agent_run.final_decision == Decision.ESCALATE.value:
        raise HTTPException(
            status_code=409,
            detail=(
                "Escalated disputes require resolution of the "
                "evidence/deadline issue before submission."
            ),
        )

    try:
        executed = execute_contest(
            _ALL_DISPUTES[dispute_id],
            agent_run,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Razorpay contest execution failed: {exc}",
        )

    executed_dict = executed.to_dict()

    # Add post-policy execution events to the visible audit ledger.
    executed_dict = _add_execution_audit(
        executed_dict,
        human_approved=True,
        executed=True,
    )

    _RUN_CACHE[dispute_id] = executed_dict

    _HUMAN_DECISIONS[dispute_id] = {
        "action": "approved_and_submitted",
        "note": req.note,
    }

    return {
        "dispute_id": dispute_id,
        "recorded": _HUMAN_DECISIONS[dispute_id],
        "action_result": executed.action_result,
    }


# ---------------------------------------------------------------------------
# Explicit AUTO_SUBMIT execution endpoint
# ---------------------------------------------------------------------------

@app.post("/disputes/{dispute_id}/execute")
def execute_authorized_dispute(dispute_id: str):
    """
    Explicit demo/test trigger for an AUTO_SUBMIT policy decision.
    """

    if dispute_id not in _ALL_DISPUTES:
        raise HTTPException(
            status_code=404,
            detail=f"Dispute {dispute_id} not found",
        )

    run = _get_or_run(dispute_id)

    if run["final_decision"] != Decision.AUTO_SUBMIT.value:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Policy decision is {run['final_decision']}, "
                "not AUTO_SUBMIT"
            ),
        )

    try:
        agent_run = run_agent(
            _ALL_DISPUTES[dispute_id],
            include_rebuttal=True,
        )

        executed = execute_contest(
            _ALL_DISPUTES[dispute_id],
            agent_run,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Razorpay contest execution failed: {exc}",
        )

    executed_dict = executed.to_dict()

    executed_dict = _add_execution_audit(
        executed_dict,
        executed=True,
    )

    _RUN_CACHE[dispute_id] = executed_dict

    return {
        "dispute_id": dispute_id,
        "action_result": executed.action_result,
    }


# ---------------------------------------------------------------------------
# Fast dashboard summary
# ---------------------------------------------------------------------------

@app.get("/dashboard")
def dashboard():
    """
    Instant dashboard summary from the bundled snapshot.

    Dynamic human approvals from the current server session are layered
    over the static policy snapshot.
    """

    if not _DASHBOARD_SNAPSHOT:
        return stats()

    counts = dict(
        _DASHBOARD_SNAPSHOT.get("decision_counts", {})
    )

    total = int(
        _DASHBOARD_SNAPSHOT.get(
            "total_disputes",
            len(_ALL_DISPUTES),
        )
    )

    risk = float(
        _DASHBOARD_SNAPSHOT.get(
            "money_at_risk",
            0.0,
        )
    )

    submitted = 0.0

    for dispute_id, human in _HUMAN_DECISIONS.items():
        if human.get("action") == "approved_and_submitted":
            submitted += float(
                _ALL_DISPUTES[dispute_id]["amount"]
            )

    pending_ids = [
        dispute_id
        for dispute_id, item in _DASHBOARD_ITEMS.items()
        if (
            item["decision"] == Decision.DRAFT_FOR_REVIEW.value
            and dispute_id not in _HUMAN_DECISIONS
        )
    ]

    pending_money = sum(
        float(_ALL_DISPUTES[dispute_id]["amount"])
        for dispute_id in pending_ids
    )

    return {
        "total_disputes": total,
        "decision_counts": counts,
        "money_at_risk": round(risk, 2),
        "money_submitted_for_defense": round(submitted, 2),
        "money_pending_human_review": round(pending_money, 2),
        "money_successfully_defended": 0.0,
        "auto_defense_rate": round(
            counts.get(Decision.AUTO_SUBMIT.value, 0) / total,
            4,
        ) if total else 0,
        "executed_submission_rate": round(
            submitted / risk,
            4,
        ) if risk else 0,
        "human_reviews_pending": len(pending_ids),
        "snapshot": True,
    }


# ---------------------------------------------------------------------------
# Stats endpoint
# ---------------------------------------------------------------------------

@app.get("/stats")
def stats():
    """
    Fast dashboard metrics.

    Uses the precomputed snapshot instead of running the full agent across
    all 10,000 disputes on every dashboard load.
    """

    if _DASHBOARD_SNAPSHOT:
        counts = dict(
            _DASHBOARD_SNAPSHOT.get("decision_counts", {})
        )

        total = int(
            _DASHBOARD_SNAPSHOT.get(
                "total_disputes",
                len(_ALL_DISPUTES),
            )
        )

        money_at_risk = float(
            _DASHBOARD_SNAPSHOT.get(
                "money_at_risk",
                0.0,
            )
        )

        money_submitted = 0.0

        for dispute_id, human in _HUMAN_DECISIONS.items():
            if human.get("action") == "approved_and_submitted":
                money_submitted += float(
                    _ALL_DISPUTES[dispute_id]["amount"]
                )

        pending_ids = [
            dispute_id
            for dispute_id, item in _DASHBOARD_ITEMS.items()
            if (
                item["decision"] == Decision.DRAFT_FOR_REVIEW.value
                and dispute_id not in _HUMAN_DECISIONS
            )
        ]

        money_pending = sum(
            float(_ALL_DISPUTES[dispute_id]["amount"])
            for dispute_id in pending_ids
        )

        return {
            "total_disputes": total,
            "decision_counts": counts,
            "money_at_risk": round(money_at_risk, 2),
            "money_submitted_for_defense": round(
                money_submitted,
                2,
            ),
            "money_pending_human_review": round(
                money_pending,
                2,
            ),
            "money_successfully_defended": 0.0,
            "auto_defense_rate": round(
                counts.get(Decision.AUTO_SUBMIT.value, 0) / total,
                4,
            ) if total else 0,
            "executed_submission_rate": round(
                money_submitted / money_at_risk,
                4,
            ) if money_at_risk else 0,
            "human_reviews_pending": len(pending_ids),
        }

    # Safe fallback if snapshot is missing.
    counts = {}
    money_at_risk = 0.0
    money_submitted = 0.0
    money_pending = 0.0
    money_successfully_defended = 0.0

    for dispute_id, dispute in _ALL_DISPUTES.items():
        run = _get_or_run(dispute_id)

        decision = run["final_decision"]

        counts[decision] = counts.get(decision, 0) + 1
        money_at_risk += float(dispute["amount"])

        result = run.get("action_result") or {}

        if result.get("executed"):
            money_submitted += float(dispute["amount"])

            status = (
                result.get("response") or {}
            ).get("status")

            if status == "won":
                money_successfully_defended += float(
                    dispute["amount"]
                )

        elif decision == Decision.DRAFT_FOR_REVIEW.value:
            money_pending += float(dispute["amount"])

    total = len(_ALL_DISPUTES)

    return {
        "total_disputes": total,
        "decision_counts": counts,
        "money_at_risk": round(
            money_at_risk,
            2,
        ),
        "money_submitted_for_defense": round(
            money_submitted,
            2,
        ),
        "money_pending_human_review": round(
            money_pending,
            2,
        ),
        "money_successfully_defended": round(
            money_successfully_defended,
            2,
        ),
        "auto_defense_rate": round(
            counts.get(Decision.AUTO_SUBMIT.value, 0) / total,
            4,
        ) if total else 0,
        "executed_submission_rate": round(
            money_submitted / money_at_risk,
            4,
        ) if money_at_risk else 0,
        "human_reviews_pending": sum(
            1
            for dispute_id in _ALL_DISPUTES
            if (
                dispute_id not in _HUMAN_DECISIONS
                and _get_or_run(dispute_id)["final_decision"]
                == Decision.DRAFT_FOR_REVIEW.value
            )
        ),
    }


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def _add_execution_audit(
    run_dict: dict,
    *,
    human_approved: bool = False,
    rejected: bool = False,
    executed: bool = False,
) -> dict:
    """
    Append execution-side events to the visible audit trail.

    The core agent records the reasoning pipeline.
    This helper records what happened after the policy decision crossed
    the execution boundary.
    """

    audit = run_dict.setdefault(
        "audit_trail",
        [],
    )

    existing_states = {
        entry.get("state")
        for entry in audit
        if isinstance(entry, dict)
    }

    def add_event(
        state: str,
        detail: str,
    ):
        if state in existing_states:
            return

        audit.append(
            {
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "state": state,
                "detail": detail,
            }
        )

        existing_states.add(state)

    if rejected:
        add_event(
            "HUMAN_REJECTED",
            "Merchant rejected the proposed dispute defense.",
        )

        add_event(
            "AUDIT_LOG",
            "Human rejection recorded.",
        )

        return run_dict

    if human_approved:
        add_event(
            "HUMAN_APPROVED",
            "Merchant approved the proposed dispute defense.",
        )

    if executed:
        result = run_dict.get(
            "action_result"
        ) or {}

        document_id = result.get(
            "document_id"
        )

        add_event(
            "EVIDENCE_PACKET_CREATED",
            "Grounded evidence packet prepared for dispute submission.",
        )

        if document_id:
            add_event(
                "DOCUMENT_UPLOADED",
                f"Evidence document uploaded successfully: {document_id}",
            )
        else:
            add_event(
                "DOCUMENT_UPLOADED",
                "Evidence document uploaded successfully.",
            )

        response = result.get(
            "response"
        ) or {}

        status = response.get(
            "status",
            "submitted",
        )

        adapter = result.get(
            "adapter",
            "configured Razorpay adapter",
        )

        add_event(
            "CONTEST_SUBMITTED",
            f"Dispute contest submitted through {adapter}; status={status}.",
        )

        add_event(
            "AUDIT_LOG",
            "Execution record finalized.",
        )

    return run_dict


# ---------------------------------------------------------------------------
# Cached agent execution
# ---------------------------------------------------------------------------

def _get_or_run(dispute_id: str) -> dict:
    if dispute_id not in _RUN_CACHE:
        run = run_agent(
            _ALL_DISPUTES[dispute_id],
            include_rebuttal=False,
        )

        _RUN_CACHE[dispute_id] = run.to_dict()

    return _RUN_CACHE[dispute_id]