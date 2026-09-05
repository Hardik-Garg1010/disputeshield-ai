# Evaluation

## Dataset

The benchmark contains **10,000 synthetic `product_not_received` disputes**, generated with correlated evidence features plus injected outcome noise. The generator is deterministic for a fixed seed.

Split: **70% train / 15% validation / 15% test**. The validation split selects the model; the test split is held back for the final reported metrics.

The benchmark is synthetic and is not a proxy for real Razorpay dispute-win rates.

## 1. Model selection

Two calibrated classifiers are trained on the same features:

- Logistic Regression + 5-fold isotonic calibration
- XGBoost + 5-fold isotonic calibration

The model is selected by **validation ROC-AUC**, then refit on train+validation and evaluated once on the untouched test set. The persisted model is `ml/models/contestability_model.joblib`.

The features are:

1. delivery confirmation
2. tracking availability
3. customer acknowledgement
4. delivery signature
5. refund issued
6. evidence count (actual evidence documents only)
7. previous dispute history

The exact generated metrics are stored in `ml/models/model_comparison.json` and `ml/models/metrics.json`.

## 2. Action-policy evaluation

Prediction quality is only one part of the product. The policy gate independently checks:

- contestability threshold
- evidence completeness
- required evidence
- policy conflicts
- merchant amount limit
- deadline safety buffer

The important safety distinction is:

- **AUTO_SUBMIT** = policy authorized autonomous execution; it is not counted as a successful defense until an actual contest action is executed and a final outcome is known.
- **DRAFT_FOR_REVIEW** = evidence/reasoning is strong enough to prepare a case, but a human must approve.
- **ESCALATE** = missing evidence, deadline risk, or low confidence requires investigation.
- **DO_NOT_CONTEST** = a hard policy conflict, such as an already-issued refund, makes contesting inappropriate.

## 3. Grounding evaluation

The LLM receives only retrieved evidence passages and must return source-linked factual claims. The verifier rejects claims whose cited source was not present in the retrieved context.

This measures **source attribution correctness**, not full semantic entailment. Therefore we do **not** claim that a current `0` unsupported-source count proves zero hallucinations. A future improvement would add claim-to-passage entailment checking.

## 4. Razorpay execution

The project uses a Razorpay adapter with two modes:

- `RazorpayLiveClient`: Test Mode credentials only.
- `RazorpaySimulator`: offline fallback, with every response explicitly marked `simulated: true`.

The API is side-effect free when merely listing or viewing disputes. Contest execution happens only through an explicit webhook/approval/execute trigger. Evidence is packaged as a PDF and uploaded with the `dispute_evidence` purpose before the contest call.

Razorpay's current API requires `action=submit` to actually submit a contest; a draft does not auto-submit. It also requires at least one evidence document for a successful submission.

## 5. Limitations

- Synthetic data only.
- One narrow dispute reason code.
- No production database.
- No multi-seed uncertainty analysis yet.
- Source matching is not semantic entailment.
- Test-mode dispute execution depends on what the Razorpay account exposes; the simulator keeps the demo reproducible when live test data is unavailable.
