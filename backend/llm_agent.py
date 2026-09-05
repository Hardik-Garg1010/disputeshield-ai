"""
DisputeShield AI - LLM Agent Layer (Milestone 4/agent, real version)

This replaces the deterministic-template `generate_rebuttal()` with an
actual LLM call, addressing the single biggest gap from the review:
"the agent isn't actually using an LLM."

Supports TWO providers, since Anthropic's API has no permanent free tier
(only a one-time ~$5 trial credit) while Google's Gemini API via AI Studio
has a genuinely permanent free tier (no card, no expiration) for the Flash
models -- pick whichever you have a key for:

    ANTHROPIC_API_KEY set  -> uses Claude (anthropic SDK)
    GEMINI_API_KEY set     -> uses Gemini (google-genai SDK)
    both set                -> Anthropic is preferred (set
                                DISPUTESHIELD_LLM_PROVIDER=gemini to force
                                Gemini instead)
    neither set              -> llm_available() returns False, agent.py
                                falls back to the deterministic template

Design (same regardless of provider):

  RAG retrieval (rag/retriever.py)
        |
        v
  Retrieved evidence passages, each with an exact source path
        |
        v
  LLM reasons over ONLY those passages, must cite a source for every
  factual claim it makes
        |
        v
  GROUNDING VERIFICATION (this file, verify_grounding()) -- for every
  claim the LLM returns, this is a TWO-stage check, not just one:
    1. Source-exists: the cited source path must be one of the sources
       the LLM was actually given (catches invented/wrong source paths).
    2. Claim-supported: the claim's own content must actually be backed
       by that source's text, and must not contradict something the
       source explicitly negates (catches the LLM citing a REAL source
       but asserting something that source doesn't say, or says the
       opposite of -- e.g. citing tracking.json but claiming a signature
       exists when the source says "no signature on file"). Stage 1
       alone cannot catch this; a real source can still be misquoted.
  Any claim failing either stage is DROPPED before the rebuttal is shown.
        |
        v
  Grounded rebuttal + real unsupported-claim count (checked, not just
  assumed by construction like the old template approach)
"""

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai as google_genai
except ImportError:
    google_genai = None

ANTHROPIC_MODEL = os.environ.get("DISPUTESHIELD_ANTHROPIC_MODEL", "").strip()
GEMINI_MODEL = os.environ.get("DISPUTESHIELD_GEMINI_MODEL", "gemini-3.7-flash").strip()

SYSTEM_PROMPT = """You are an evidence-grounded dispute-rebuttal drafter for a payments company.

You will be given a dispute and a list of retrieved EVIDENCE PASSAGES, each with an exact source path.

Your job: write a short, professional dispute-representment rebuttal, but you may ONLY state facts that are directly supported by one of the given evidence passages. For every factual claim, you must cite the EXACT source path it came from -- copy it character-for-character from the evidence list, never invent or guess a source path.

If the evidence does not support contesting the dispute at all, say so plainly instead of inventing supporting claims.

Respond with ONLY valid JSON in this exact shape, nothing else, no markdown fences:
{
  "claims": [
    {"claim": "short factual sentence", "source": "exact source path from the evidence list"}
  ],
  "summary": "one paragraph rebuttal summary in professional dispute-representment tone, referencing only the claims above"
}
"""


def build_user_prompt(dispute: dict, retrieved_evidence: list) -> str:
    evidence_block = "\n".join(
        f"- [{e['source']}] {e['text']}" for e in retrieved_evidence
    )
    return (
        f"Dispute ID: {dispute['dispute_id']}\n"
        f"Amount: {dispute['amount']} {dispute.get('currency', 'INR')}\n"
        f"Reason: {dispute['reason_code']}\n\n"
        f"EVIDENCE PASSAGES (only cite these exact source paths):\n{evidence_block}\n"
    )


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "was", "were", "is", "are", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "by",
    "from", "as", "that", "this", "it", "its", "their", "there", "which",
    "who", "we", "request", "dispute", "order", "attached", "based",
}

# A crude 4-char prefix stem -- not real stemming, but enough to match
# "signed"/"signature" or "delivered"/"delivery" against each other without
# pulling in a full NLP dependency for what is fundamentally a lightweight
# sanity check, not an NLI model.
def _stem(word: str) -> str:
    return word[:4] if len(word) >= 4 else word


def _content_stems(text: str) -> set:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {_stem(w) for w in words if w not in _STOPWORDS}


# Matches "no signature", "not confirmed", "without a receipt", etc. so we
# can tell when the SOURCE itself is denying something -- this is what
# catches a claim that cites a real source but asserts the opposite of
# what that source says.
_NEGATION_PATTERN = re.compile(r"\b(?:no|not|without|never|isn't|wasn't)\s+([a-zA-Z]{3,})")


def _negated_stems(text: str) -> set:
    return {_stem(m.group(1)) for m in _NEGATION_PATTERN.finditer(text.lower())}


def _claim_supported_by_source(claim: str, source_text: str, min_overlap: float = 0.5):
    """Lightweight lexical entailment check. This is deliberately NOT a
    full NLI model -- it's a real second gate beyond 'did the LLM cite a
    source it was given', catching the two failure modes a source-exists
    check misses entirely:

      1. Off-topic: the claim's content words barely appear in the cited
         source at all (the LLM cited a real source but wrote about
         something else).
      2. Contradiction: the source explicitly negates a concept ("no
         signature on file") that the claim asserts positively ("the
         customer signed for the package") -- citing a real, relevant
         source while still saying the opposite of what it states.

    Returns (supported: bool, reason: str).
    """
    claim_stems = _content_stems(claim)
    if not claim_stems:
        return False, "claim has no checkable content"

    negated_in_source = _negated_stems(source_text)
    contradicted = claim_stems & negated_in_source
    if contradicted:
        return False, f"source explicitly negates: {', '.join(sorted(contradicted))}"

    source_stems = _content_stems(source_text)
    overlap = claim_stems & source_stems
    overlap_ratio = len(overlap) / len(claim_stems)
    if overlap_ratio < min_overlap:
        return False, (f"insufficient lexical support "
                        f"({overlap_ratio:.0%} of claim content found in source)")

    return True, "supported"


def verify_grounding(claims: list, retrieved_evidence: list) -> tuple:
    """
    The actual grounding check, in two stages (see the module docstring):
    source-exists, then claim-supported-by-that-source's-actual-text. This
    is what makes 'unsupported claim rate' a real, tested number instead
    of an assumption -- we are checking the model's output against ground
    truth, not just checking it name-dropped a real filename.

    Returns (grounded_claims, unsupported_count, rejected_claims), where
    rejected_claims carries the reason each dropped claim failed, for the
    audit trail / "why was this claim rejected" UI.
    """
    source_text_by_path = {e["source"]: e["text"] for e in retrieved_evidence}
    grounded = []
    rejected = []

    for c in claims:
        source = c.get("source", "")
        claim_text = c.get("claim", "")

        if source not in source_text_by_path or not claim_text:
            rejected.append({"claim": claim_text, "source": source,
                              "reason": "cited source was not in the retrieved evidence set"})
            continue

        supported, reason = _claim_supported_by_source(claim_text, source_text_by_path[source])
        if supported:
            grounded.append(c)
        else:
            rejected.append({"claim": claim_text, "source": source, "reason": reason})

    return grounded, len(rejected), rejected


def _active_provider() -> str:
    forced = os.environ.get("DISPUTESHIELD_LLM_PROVIDER", "").lower()
    if forced in ("anthropic", "gemini"):
        return forced
    if os.environ.get("ANTHROPIC_API_KEY") and anthropic is not None:
        return "anthropic"
    if os.environ.get("GEMINI_API_KEY") and google_genai is not None:
        return "gemini"
    return "none"


def _call_anthropic(user_prompt: str) -> str:
    if not ANTHROPIC_MODEL:
        raise RuntimeError("DISPUTESHIELD_ANTHROPIC_MODEL is not set")
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def _call_gemini(user_prompt: str) -> str:
    if not GEMINI_MODEL:
        raise RuntimeError("DISPUTESHIELD_GEMINI_MODEL is not set")
    client = google_genai.Client()  # reads GEMINI_API_KEY from env
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    return response.text.strip()


def generate_rebuttal_llm(dispute: dict, retrieved_evidence: list) -> dict:
    """
    Real LLM call + grounding verification, via whichever provider is
    configured (see _active_provider()). Returns the same shape as the
    old template generator so agent.py doesn't need to change how it
    consumes the result:
        {"body": str, "grounded_claims": [...], "unsupported_claim_count": int}
    """
    provider = _active_provider()
    if provider == "none":
        raise RuntimeError("No LLM configured -- set ANTHROPIC_API_KEY or GEMINI_API_KEY")

    user_prompt = build_user_prompt(dispute, retrieved_evidence)

    if provider == "anthropic":
        raw_text = _call_anthropic(user_prompt)
        model_label = f"anthropic:{ANTHROPIC_MODEL}"
    else:
        raw_text = _call_gemini(user_prompt)
        model_label = f"gemini:{GEMINI_MODEL}"

    # Defensive parsing: strip markdown fences if the model adds them
    # despite instructions not to.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    parsed = json.loads(raw_text)

    llm_claims = parsed.get("claims", [])
    grounded_claims, unsupported_count, rejected_claims = verify_grounding(llm_claims, retrieved_evidence)

    if not grounded_claims:
        body = (
            "Insufficient grounded evidence exists to support a rebuttal. "
            "No submission was drafted; escalate to manual investigation if needed."
        )
    else:
        claim_lines = "\n".join(f"- {c['claim']} (source: {c['source']})" for c in grounded_claims)
        body = (
            f"DISPUTE REPRESENTMENT\n\n"
            f"Dispute: {dispute['dispute_id']}  Amount: {dispute['amount']} {dispute.get('currency', 'INR')}\n"
            f"Reason: {dispute['reason_code']}\n\n"
            f"{parsed.get('summary', '')}\n\n"
            f"Supporting evidence:\n{claim_lines}\n"
        )

    return {
        "body": body,
        "grounded_claims": grounded_claims,
        "unsupported_claim_count": unsupported_count,
        "rejected_claims": rejected_claims,  # each with a "reason" -- feeds the audit trail / "why was this claim rejected" view
        "llm_model": model_label,
    }


def llm_available() -> bool:
    return _active_provider() != "none"


if __name__ == "__main__":
    # Smoke test with a made-up dispute + evidence, to check the plumbing
    # without depending on the full retriever/dataset.
    provider = _active_provider()
    if provider == "none":
        print("No LLM configured -- set ANTHROPIC_API_KEY or GEMINI_API_KEY "
              "and re-run to actually test a live call.")
    else:
        print(f"Using provider: {provider}")
        dispute = {"dispute_id": "DSP_TEST", "amount": 4999.0,
                    "currency": "INR", "reason_code": "product_not_received"}
        evidence = [
            {"source": "synthetic_data/orders/ORD_TEST.json",
             "text": "Order ORD_TEST: status delivered, amount INR 4999."},
            {"source": "synthetic_data/shipping/ORD_TEST_tracking.json",
             "text": "Tracking status: delivered, signature on file."},
        ]
        result = generate_rebuttal_llm(dispute, evidence)
        print(json.dumps(result, indent=2))