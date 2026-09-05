# 🛡️ DisputeShield AI

### Evidence-grounded, policy-gated dispute defense for Razorpay merchants

Built for the **Razorpay AI Buildathon 2026 — AI Risk Manager Track**.

> **Detect → Retrieve → Score → Reason → Gate → Draft/Submit → Audit**

DisputeShield AI is an autonomous chargeback-defense agent designed to help merchants respond to payment disputes faster while preventing unsafe or unsupported submissions.

It combines:

- 🔎 **RAG-based evidence retrieval**
- 🧠 **Calibrated ML contestability scoring**
- ✍️ **Grounded LLM rebuttal generation**
- 🛡️ **Deterministic policy gating**
- 👤 **Human-in-the-loop approval**
- 📄 **Automated evidence packet generation**
- 💳 **Razorpay contest API adapter**
- 📋 **Complete execution audit trail**

The system is intentionally designed so that **the LLM never has unrestricted authority to submit a dispute**.

---

# 1. The Problem

When a merchant receives a chargeback or payment dispute, the challenge is not simply deciding whether to contest it.

The merchant must:

1. Understand the dispute.
2. Retrieve relevant order and delivery evidence.
3. Determine whether the evidence actually supports a defense.
4. Prepare a clear and defensible response.
5. Respect the response deadline.
6. Avoid submitting cases with missing or conflicting evidence.
7. Track what was submitted and why.

Manual dispute handling makes this process slow, inconsistent, and difficult to scale.

DisputeShield AI automates the evidence analysis and drafting workflow while keeping the final authorization behind deterministic safety controls.

---

# 2. Our Solution

DisputeShield AI treats every dispute as an evidence-grounded decision problem.

The agent follows:

```text
DISPUTE
   │
   ▼
DETECT
   │
   ▼
RETRIEVE EVIDENCE
   │
   ├── Order data
   ├── Shipment/tracking data
   ├── Customer communication
   └── Merchant policy
   │
   ▼
SCORE CONTESTABILITY
   │
   ▼
GENERATE GROUNDED DEFENSE
   │
   ▼
DETERMINISTIC POLICY GATE
   │
   ├── AUTO_SUBMIT
   ├── DRAFT_FOR_REVIEW
   ├── ESCALATE
   └── DO_NOT_CONTEST
   │
   ▼
EVIDENCE PACKET
   │
   ▼
RAZORPAY CONTEST API
   │
   ▼
AUDIT TRAIL