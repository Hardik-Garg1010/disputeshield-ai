import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from policy_engine import Decision, PolicyConfig, evaluate_policy


class PolicyEngineTests(unittest.TestCase):
    def test_strong_case_auto_submit(self):
        evidence = {
            "delivery_confirmed": True,
            "tracking_available": True,
            "customer_acknowledged_delivery": True,
            "delivery_signature": True,
            "refund_issued": False,
        }
        result = evaluate_policy(0.95, evidence, 5000, 10)
        self.assertEqual(result.decision, Decision.AUTO_SUBMIT)

    def test_missing_required_evidence_escalates(self):
        evidence = {
            "delivery_confirmed": False,
            "tracking_available": False,
            "customer_acknowledged_delivery": False,
            "delivery_signature": False,
            "refund_issued": False,
        }
        result = evaluate_policy(0.95, evidence, 5000, 10)
        self.assertEqual(result.decision, Decision.ESCALATE)

    def test_refund_conflict_is_hard_block(self):
        evidence = {
            "delivery_confirmed": True,
            "tracking_available": True,
            "customer_acknowledged_delivery": True,
            "delivery_signature": True,
            "refund_issued": True,
        }
        result = evaluate_policy(0.99, evidence, 5000, 10)
        self.assertEqual(result.decision, Decision.DO_NOT_CONTEST)

    def test_deadline_risk_escalates(self):
        evidence = {
            "delivery_confirmed": True,
            "tracking_available": True,
            "customer_acknowledged_delivery": True,
            "delivery_signature": True,
            "refund_issued": False,
        }
        result = evaluate_policy(0.95, evidence, 5000, 1)
        self.assertEqual(result.decision, Decision.ESCALATE)

    def test_draft_thresholds_are_configurable(self):
        """A mid-confidence case that would normally hit DRAFT_FOR_REVIEW
        under the default 0.60/0.50 draft thresholds should ESCALATE
        instead once a merchant configures stricter draft thresholds --
        proving draft routing isn't hardcoded and independently follows
        PolicyConfig like the auto-submit thresholds do."""
        evidence = {
            "delivery_confirmed": True,
            "tracking_available": True,
            "customer_acknowledged_delivery": False,
            "delivery_signature": False,
            "refund_issued": False,
        }
        default_result = evaluate_policy(0.65, evidence, 5000, 10)
        self.assertEqual(default_result.decision, Decision.DRAFT_FOR_REVIEW)

        strict_config = PolicyConfig(
            draft_contestability_threshold=0.90,
            draft_completeness_threshold=0.90,
        )
        strict_result = evaluate_policy(0.65, evidence, 5000, 10, config=strict_config)
        self.assertEqual(strict_result.decision, Decision.ESCALATE)


if __name__ == "__main__":
    unittest.main()