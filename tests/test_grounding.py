import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from llm_agent import verify_grounding


TRACKING_EVIDENCE = {
    "source": "synthetic_data/shipping/ORD_00021_tracking.json",
    "text": ("Shipment for order ORD_00021 via BlueDart, tracking number TRK123456789, "
             "status: delivered, with no signature on file, delivery date 2026-06-15."),
}

ORDER_EVIDENCE = {
    "source": "synthetic_data/orders/ORD_00021.json",
    "text": ("Order ORD_00021: Wireless Earbuds Pro, amount INR 4999, "
             "ordered on 2026-06-01, current status: delivered."),
}

CHAT_EVIDENCE = {
    "source": "synthetic_data/communications/ORD_00021_chat.txt",
    "text": ("Customer: Hi, checking on order ORD_00021\n"
             "Support: It shows delivered on 2026-06-15. Did you receive it?\n"
             "Customer: Yes got it, thanks!\n"),
}

EVIDENCE = [TRACKING_EVIDENCE, ORDER_EVIDENCE, CHAT_EVIDENCE]


class GroundingVerificationTests(unittest.TestCase):
    def test_supported_claim_is_kept(self):
        claims = [{"claim": "The order was delivered on June 15.", "source": ORDER_EVIDENCE["source"]}]
        grounded, unsupported, rejected = verify_grounding(claims, EVIDENCE)
        self.assertEqual(len(grounded), 1)
        self.assertEqual(unsupported, 0)
        self.assertEqual(rejected, [])

    def test_hallucinated_source_is_rejected(self):
        """Stage 1: citing a source that was never retrieved at all."""
        claims = [{"claim": "The customer signed for the package.",
                   "source": "synthetic_data/shipping/ORD_99999_tracking.json"}]
        grounded, unsupported, rejected = verify_grounding(claims, EVIDENCE)
        self.assertEqual(grounded, [])
        self.assertEqual(unsupported, 1)
        self.assertIn("not in the retrieved evidence set", rejected[0]["reason"])

    def test_contradicting_a_real_source_is_rejected(self):
        """Stage 2: the exact GPT-review failure mode -- a REAL, retrieved
        source is cited, but the claim asserts the opposite of what that
        source actually says (source: 'no signature on file';
        claim: customer signed for it)."""
        claims = [{"claim": "The customer personally signed for the package.",
                   "source": TRACKING_EVIDENCE["source"]}]
        grounded, unsupported, rejected = verify_grounding(claims, EVIDENCE)
        self.assertEqual(grounded, [])
        self.assertEqual(unsupported, 1)
        self.assertIn("negates", rejected[0]["reason"])

    def test_off_topic_claim_citing_real_source_is_rejected(self):
        """A real source is cited, but the claim is about something that
        source never discusses at all (low lexical overlap)."""
        claims = [{"claim": "The customer requested a full refund via email.",
                   "source": CHAT_EVIDENCE["source"]}]
        grounded, unsupported, rejected = verify_grounding(claims, EVIDENCE)
        self.assertEqual(grounded, [])
        self.assertEqual(unsupported, 1)
        self.assertIn("insufficient lexical support", rejected[0]["reason"])

    def test_no_source_at_all_is_rejected(self):
        claims = [{"claim": "The order was delivered.", "source": ""}]
        grounded, unsupported, rejected = verify_grounding(claims, EVIDENCE)
        self.assertEqual(grounded, [])
        self.assertEqual(unsupported, 1)


if __name__ == "__main__":
    unittest.main()