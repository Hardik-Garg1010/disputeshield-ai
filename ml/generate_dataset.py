"""
DisputeShield AI - Synthetic Dispute Dataset Generator (Milestone 1, v2)

Changes from v1 (per review feedback):
  - Removed all uses of Python's built-in hash() for anything that needs to
    be reproducible across runs/processes. hash() on strings is randomized
    per-process by default (PYTHONHASHSEED), so "same seed" did NOT
    guarantee "same generated evidence" before this fix. Deterministic
    values are now derived from the seeded `rng` or from a fixed SHA-256
    digest, never from hash().
  - random.choice() (module-level, global RNG) replaced with rng.choice()
    (the seeded RNG) for carrier selection -- it was silently bypassing
    the seed before.
  - evidence_count now only counts ACTUAL EVIDENCE DOCUMENTS (delivery,
    tracking, acknowledgement, signature) -- it no longer folds in
    "refund_issued == False" as if the absence of a refund were itself a
    piece of evidence. That was conflating a transaction-state fact with
    a retrieved evidence source. refund_issued is still a real, important
    feature for scoring -- it's just tracked separately now.

Usage:
    python generate_dataset.py --n 10000 --seed 42
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
import random

REASON_CODE = "product_not_received"


def rand_bool(p_true: float, rng: random.Random) -> bool:
    return rng.random() < p_true


def deterministic_int(key: str, mod: int) -> int:
    """Replaces hash(key) % mod with a value that's stable across runs and
    Python processes (hash() on str is salted per-process by default)."""
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest[:12], 16) % mod


def make_dispute(i: int, rng: random.Random):
    """
    Build one synthetic dispute with correlated features.

    Strategy: sample a hidden "true case strength" in [0,1], then sample
    observed features so they correlate with it but aren't perfectly
    deterministic (noise). The label (contest_won) is then a noisy function
    of case strength, so the classifier has to learn from features, not
    just copy one field.
    """
    case_strength = rng.betavariate(2, 2)  # spread across [0,1], not clumped at extremes

    delivery_confirmed = rand_bool(0.15 + 0.75 * case_strength, rng)
    tracking_available = rand_bool(0.20 + 0.70 * case_strength, rng)
    customer_acknowledged = rand_bool(0.05 + 0.55 * case_strength * (1 if delivery_confirmed else 0.3), rng)
    # A delivery signature can only exist if delivery was actually confirmed
    # AND tracking exists to record it -- otherwise we'd generate logically
    # impossible evidence (signature on file for a shipment still in transit).
    delivery_signature = (
        delivery_confirmed and tracking_available
        and rand_bool(0.15 + 0.6 * case_strength, rng)
    )
    refund_requested = rand_bool(0.4 - 0.2 * case_strength, rng)
    refund_issued = refund_requested and rand_bool(0.3, rng)

    amount = round(rng.choice([
        rng.uniform(300, 2000),
        rng.uniform(2000, 10000),
        rng.uniform(10000, 30000),
    ]), 2)

    customer_order_count = rng.choices([1, 2, 5, 15, 40], weights=[30, 25, 25, 15, 5])[0]
    customer_previous_disputes = rng.choices([0, 1, 2, 5], weights=[70, 18, 8, 4])[0]

    days_since_delivery = rng.randint(1, 45) if delivery_confirmed else rng.randint(5, 60)
    respond_by_days = rng.randint(3, 14)

    # evidence_count: ONLY real retrieved evidence documents. Transaction
    # state (refund_issued) is tracked as its own feature below, not folded
    # in here -- "no refund happened" is not a document you can point to.
    evidence_flags = [delivery_confirmed, tracking_available,
                       customer_acknowledged, delivery_signature]
    evidence_count = sum(1 for f in evidence_flags if f)

    # Noisy label: real outcome depends on case_strength + some of the
    # observed features, plus a random flip to simulate real-world noise
    # (network/arbiter unpredictability, edge cases).
    win_prob = (
        0.05
        + 0.55 * case_strength
        + 0.15 * (1 if delivery_confirmed else 0)
        + 0.10 * (1 if customer_acknowledged else 0)
        + 0.05 * (1 if tracking_available else 0)
        - 0.15 * (1 if refund_issued else 0)
    )
    win_prob = min(max(win_prob, 0.02), 0.98)
    contest_won = rand_bool(win_prob, rng)

    dispute_id = f"DSP_{i:05d}"
    # Deterministic pseudo-random payment id from the seeded RNG, not uuid4
    # (uuid4 is fine for uniqueness but isn't reproducible run-to-run; using
    # the seeded rng keeps the WHOLE dataset reproducible from --seed alone).
    payment_id = f"pay_{rng.getrandbits(56):014x}"
    customer_id = f"CUST_{rng.randint(1, 300):04d}"
    order_id = f"ORD_{i:05d}"

    return {
        "dispute_id": dispute_id,
        "payment_id": payment_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "amount": amount,
        "currency": "INR",
        "reason_code": REASON_CODE,
        "phase": rng.choice(["under_review", "action_required"]),
        "respond_by_days": respond_by_days,
        "days_since_delivery": days_since_delivery,
        # Evidence features (real, retrievable documents)
        "delivery_confirmed": delivery_confirmed,
        "tracking_available": tracking_available,
        "customer_acknowledged_delivery": customer_acknowledged,
        "delivery_signature": delivery_signature,
        "evidence_count": evidence_count,
        # Transaction-state features (facts about the transaction, not
        # documents -- kept separate from evidence_count on purpose)
        "refund_requested": refund_requested,
        "refund_issued": refund_issued,
        "customer_order_count": customer_order_count,
        "customer_previous_disputes": customer_previous_disputes,
        # Label (only used for training/eval, not shown to the "live" agent)
        "contest_won": contest_won,
        "case_strength_hidden": round(case_strength, 3),
    }


def make_evidence_docs(dispute: dict, out_dir: Path, rng: random.Random):
    """Write per-dispute evidence documents into the RAG folder structure.
    All derived values use `rng` or deterministic_int() -- never hash()."""
    d = dispute
    order_date = (datetime(2026, 6, 1) +
                  timedelta(days=deterministic_int(d["order_id"] + "-date", 60))).strftime("%Y-%m-%d")

    order = {
        "order_id": d["order_id"],
        "customer_id": d["customer_id"],
        "amount": d["amount"],
        "currency": d["currency"],
        "product": "Wireless Earbuds Pro" if d["amount"] < 3000 else "Smart Home Hub",
        "order_date": order_date,
        "status": "delivered" if d["delivery_confirmed"] else "shipped",
    }
    (out_dir / "orders" / f"{d['order_id']}.json").write_text(json.dumps(order, indent=2))

    if d["tracking_available"]:
        tracking = {
            "order_id": d["order_id"],
            "carrier": rng.choice(["BlueDart", "Delhivery", "Ekart"]),
            "tracking_number": f"TRK{deterministic_int(d['order_id'] + '-track', 10**9)}",
            "status": "delivered" if d["delivery_confirmed"] else "in_transit",
            "delivered_signature": d["delivery_signature"],  # gated on
            # delivery_confirmed in make_dispute(); a signature can only be
            # true here if status above is "delivered" too.
            "delivered_date": order_date,
        }
        (out_dir / "shipping" / f"{d['order_id']}_tracking.json").write_text(json.dumps(tracking, indent=2))

    if d["customer_acknowledged_delivery"]:
        chat = (
            f"Customer: Hi, checking on order {d['order_id']}\n"
            f"Support: It shows delivered on {order_date}. Did you receive it?\n"
            f"Customer: Yes got it, thanks!\n"
        )
        (out_dir / "communications" / f"{d['order_id']}_chat.txt").write_text(chat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000, help="number of synthetic disputes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/disputes.jsonl")
    ap.add_argument("--emit-evidence", action="store_true", default=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_root = Path(__file__).parent.parent / "synthetic_data"
    for sub in ["orders", "shipping", "communications", "policies"]:
        (evidence_root / sub).mkdir(parents=True, exist_ok=True)

    policy_md = (
        "# Refund & Delivery Dispute Policy\n\n"
        "- Orders are considered delivered once carrier tracking shows `delivered` status.\n"
        "- If a customer acknowledges receipt in chat, this overrides a delivery dispute.\n"
        "- Refunds already issued cannot be re-charged; do not contest disputes where "
        "refund_issued = true.\n"
    )
    (evidence_root / "policies" / "refund_policy.md").write_text(policy_md)

    disputes = []
    with open(out_path, "w") as f:
        for i in range(1, args.n + 1):
            d = make_dispute(i, rng)
            disputes.append(d)
            f.write(json.dumps(d) + "\n")
            if args.emit_evidence:
                make_evidence_docs(d, evidence_root, rng)

    won = sum(1 for d in disputes if d["contest_won"])
    print(f"Generated {len(disputes)} disputes -> {out_path}")
    print(f"Positive rate (contest_won): {won}/{len(disputes)} = {won/len(disputes):.1%}")
    print(f"Evidence docs written under: {evidence_root}")


if __name__ == "__main__":
    main()