"""DisputeShield AI - reproducible contestability model training."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURES = [
    "delivery_confirmed", "tracking_available",
    "customer_acknowledged_delivery", "delivery_signature",
    "refund_issued", "evidence_count", "customer_previous_disputes",
]

def load_dataset(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def featurize(d):
    return [
        float(bool(d["delivery_confirmed"])),
        float(bool(d["tracking_available"])),
        float(bool(d["customer_acknowledged_delivery"])),
        float(bool(d["delivery_signature"])),
        float(bool(d["refund_issued"])),
        float(d["evidence_count"]) / 4.0,
        min(float(d["customer_previous_disputes"]), 5.0) / 5.0,
    ]

def evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "test_size": int(len(y_true)),
    }

def build_models():
    lr = CalibratedClassifierCV(LogisticRegression(max_iter=2000, random_state=7), method="isotonic", cv=5)
    xgb_base = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=7, n_jobs=1,
    )
    xgb = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    return lr, xgb

def main():
    data_path = Path(__file__).parent / "data" / "disputes.jsonl"
    rows = load_dataset(data_path)
    X = np.asarray([featurize(d) for d in rows], dtype=float)
    y = np.asarray([int(d["contest_won"]) for d in rows], dtype=int)

    # 70/15/15: validation chooses the model; test is untouched until final evaluation.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=7, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=7, stratify=y_temp)

    lr, xgb = build_models()
    lr.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    lr_val = evaluate(y_val, lr.predict_proba(X_val)[:, 1])
    xgb_val = evaluate(y_val, xgb.predict_proba(X_val)[:, 1])
    chosen_name = "xgboost_calibrated" if xgb_val["roc_auc"] > lr_val["roc_auc"] else "logistic_regression_calibrated"

    # Refit the selected calibrated model on train+validation before the one-time test evaluation.
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    if chosen_name == "xgboost_calibrated":
        _, final_model = build_models()
    else:
        final_model, _ = build_models()
    final_model.fit(X_trainval, y_trainval)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob)

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(final_model, models_dir / "contestability_model.joblib")

    comparison = {
        "chosen_model": chosen_name,
        "validation": {"logistic_regression_calibrated": lr_val, "xgboost_calibrated": xgb_val},
        "final_test": test_metrics,
        "features": FEATURES,
        "split": {"train": len(y_train), "validation": len(y_val), "test": len(y_test), "total": len(y)},
        "selection_rule": "Choose by validation ROC-AUC; evaluate the selected model once on the held-out test set; probabilities are isotonic-calibrated.",
    }
    (models_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (models_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    # Human-readable LR coefficients are retained as a diagnostic artifact only.
    lr_plain = LogisticRegression(max_iter=2000, random_state=7).fit(X_trainval, y_trainval)
    coef_report = dict(zip(FEATURES, lr_plain.coef_[0].round(4).tolist()))
    coef_report["intercept"] = round(float(lr_plain.intercept_[0]), 4)
    (models_dir / "lr_coefficients.json").write_text(json.dumps(coef_report, indent=2), encoding="utf-8")

    # Remove obsolete metadata from older model versions.
    stale = models_dir / "model.json"
    if stale.exists():
        stale.unlink()

    print("=== Validation model selection ===")
    print(json.dumps(comparison["validation"], indent=2))
    print(f"Chosen model: {chosen_name}")
    print("=== Held-out test metrics ===")
    print(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()
