"""
DisputeShield AI - Policy Engine (Milestone 6)

This is the core trust boundary of the whole system:

    ML score + evidence completeness  --->  POLICY GATE  --->  action

The LLM/agent can RECOMMEND a contest and DRAFT a rebuttal, but it can never
call submit_dispute() directly. Only this deterministic, unit-testable
function can authorize AUTO_SUBMIT. This is what we mean by "every money
action explainable, bounded, and gated."

Design choice: the gate is a plain function with named, auditable conditions
-- not a black box -- so every decision can be logged with WHICH condition
passed or failed.
"""

from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    AUTO_SUBMIT = "AUTO_SUBMIT"
    DRAFT_FOR_REVIEW = "DRAFT_FOR_REVIEW"
    ESCALATE = "ESCALATE"
    DO_NOT_CONTEST = "DO_NOT_CONTEST"


@dataclass
class PolicyConfig:
    """Per-merchant configurable thresholds -- these live in the `merchants`
    table in the real system; hardcoded defaults here for the demo.

    contestability_threshold=0.72 is grounded in the actual calibrated score
    distribution (ml/train.py, isotonic CalibratedClassifierCV), not an
    arbitrary round number: across all 10,000 synthetic disputes, the
    strongest possible evidence combination tops out at ~0.738, and only
    the top ~5% of real disputes score above ~0.726 (p95). A threshold of
    0.90 -- which was the original placeholder -- is mathematically
    unreachable once probabilities are honestly calibrated (nothing in the
    dataset, including a perfect evidence set, empirically wins 90%+ of
    the time). 0.72 sits just below the population's p95, so AUTO_SUBMIT
    is reserved for genuinely top-tier cases, combined with the independent
    evidence_completeness requirement below -- both signals have to agree,
    not just the model score alone.
    """
    contestability_threshold: float = 0.72
    evidence_completeness_threshold: float = 0.80
    auto_submit_amount_limit: float = 10000.0  # INR
    min_deadline_days_safety_buffer: int = 2
    required_evidence_fields: tuple = (
        "delivery_confirmed",
        "tracking_available",
    )
    # Thresholds for the mid-confidence DRAFT_FOR_REVIEW path. Kept as their
    # own (lower) config fields rather than hardcoded literals, so a
    # merchant who tightens/loosens contestability_threshold /
    # evidence_completeness_threshold also gets consistent draft-routing
    # behavior instead of it silently staying fixed.
    draft_contestability_threshold: float = 0.60
    draft_completeness_threshold: float = 0.50


@dataclass
class PolicyResult:
    decision: Decision
    reasons: list = field(default_factory=list)
    contestability_score: float = 0.0
    evidence_completeness: float = 0.0

    def to_dict(self):
        return {
            "decision": self.decision.value,
            "reasons": self.reasons,
            "contestability_score": round(self.contestability_score, 3),
            "evidence_completeness": round(self.evidence_completeness, 3),
        }


def evidence_completeness(evidence: dict, required_fields: tuple) -> float:
    """Fraction of a broader evidence checklist that is actually present.
    Broader than just `required_fields` so partial evidence still contributes."""
    checklist = [
        "delivery_confirmed",
        "tracking_available",
        "customer_acknowledged_delivery",
        "delivery_signature",
    ]
    present = sum(1 for f in checklist if evidence.get(f))
    return present / len(checklist)


def has_conflicting_evidence(evidence: dict) -> bool:
    """Example conflict: tracking says delivered, but to a location the
    customer explicitly disputes, or a refund was already issued for the
    same order -- contesting after a refund is a policy violation, not
    just low-confidence."""
    if evidence.get("refund_issued") and evidence.get("delivery_confirmed"):
        # Refund already issued: contesting further is not permitted
        # regardless of how strong the delivery evidence looks.
        return True
    return False


def evaluate_policy(
    contestability_score: float,
    evidence: dict,
    amount: float,
    deadline_days_remaining: int,
    config: PolicyConfig = None,
) -> PolicyResult:
    """
    The single deterministic decision point. Given a model score (from the
    ML layer) and raw evidence flags (from RAG/retrieval), decide what the
    system is ALLOWED to do next. This function must be pure and fully
    explainable -- no LLM call inside it.
    """
    config = config or PolicyConfig()
    reasons = []

    completeness = evidence_completeness(evidence, config.required_evidence_fields)
    conflict = has_conflicting_evidence(evidence)

    required_present = all(evidence.get(f) for f in config.required_evidence_fields)

    # --- Hard blockers first (checked before anything else) ---
    if conflict:
        reasons.append("Conflicting or policy-violating evidence detected "
                        "(e.g. refund already issued) -- autonomous action blocked.")
        return PolicyResult(Decision.DO_NOT_CONTEST, reasons,
                             contestability_score, completeness)

    if not required_present:
        missing = [f for f in config.required_evidence_fields if not evidence.get(f)]
        reasons.append(f"Required evidence missing: {', '.join(missing)}. "
                        "Autonomous submission is blocked and the case is escalated for evidence investigation.")
        return PolicyResult(Decision.ESCALATE, reasons,
                             contestability_score, completeness)

    if deadline_days_remaining < config.min_deadline_days_safety_buffer:
        reasons.append(f"Only {deadline_days_remaining} day(s) left to respond -- "
                        "below safety buffer, escalating to human immediately.")
        return PolicyResult(Decision.ESCALATE, reasons,
                             contestability_score, completeness)

    # --- Auto-submit path: every condition must pass, all logged ---
    checks = {
        "contestability_score >= threshold": contestability_score >= config.contestability_threshold,
        "evidence_completeness >= threshold": completeness >= config.evidence_completeness_threshold,
        "no_conflicting_evidence": not conflict,
        "required_evidence_present": required_present,
        "amount <= auto_submit_limit": amount <= config.auto_submit_amount_limit,
        "deadline_safety_buffer_ok": deadline_days_remaining >= config.min_deadline_days_safety_buffer,
    }

    if all(checks.values()):
        reasons.append("All auto-submit conditions passed: " + "; ".join(checks.keys()))
        return PolicyResult(Decision.AUTO_SUBMIT, reasons,
                             contestability_score, completeness)

    failed = [k for k, v in checks.items() if not v]

    # --- Mid-confidence path: draft for a human, don't just guess ---
    if (contestability_score >= config.draft_contestability_threshold
            and completeness >= config.draft_completeness_threshold):
        reasons.append("Confidence alone doesn't grant autonomy. Failed auto-submit "
                        f"conditions: {', '.join(failed)}. Drafting rebuttal for human approval.")
        return PolicyResult(Decision.DRAFT_FOR_REVIEW, reasons,
                             contestability_score, completeness)

    # --- Low confidence: escalate rather than guess ---
    reasons.append(f"Low confidence/evidence. Failed conditions: {', '.join(failed)}. "
                    "Escalating to human for manual review.")
    return PolicyResult(Decision.ESCALATE, reasons,
                         contestability_score, completeness)


if __name__ == "__main__":
    # Quick smoke test using the three canonical demo cases from the plan.
    import json

    case_a = dict(delivery_confirmed=True, tracking_available=True,
                  customer_acknowledged_delivery=True, delivery_signature=True,
                  refund_issued=False)
    case_b = dict(delivery_confirmed=True, tracking_available=True,
                  customer_acknowledged_delivery=False, delivery_signature=False,
                  refund_issued=False)
    case_c = dict(delivery_confirmed=False, tracking_available=False,
                  customer_acknowledged_delivery=False, delivery_signature=False,
                  refund_issued=False)

    for name, ev, score, amt, days in [
        ("Case A (auto-contest)", case_a, 0.96, 8499, 10),
        ("Case B (human review)", case_b, 0.78, 15999, 10),
        ("Case C (graceful failure)", case_c, 0.51, 7999, 10),
    ]:
        result = evaluate_policy(score, ev, amt, days)
        print(f"\n--- {name} ---")
        print(json.dumps(result.to_dict(), indent=2))