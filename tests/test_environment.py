import unittest

from models import InvoiceAction
from server.invoice_environment import InvoiceEnv


class InvoiceEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = InvoiceEnv()
        self.env.reset(task="easy")

    def test_observation_exposes_categories_and_rules(self) -> None:
        obs = self.env.reset(task="medium")
        self.assertIn("office_supplies", obs.valid_categories)
        self.assertTrue(any("purchase order" in rule for rule in obs.policy_rules))

    def test_viewing_two_different_invoices_is_not_penalized(self) -> None:
        first = self.env.step(InvoiceAction(type="view_invoice", invoice_id="INV001"))
        second = self.env.step(InvoiceAction(type="view_invoice", invoice_id="INV002"))

        self.assertAlmostEqual(first.reward, 0.08, places=2)
        self.assertAlmostEqual(second.reward, 0.08, places=2)
        self.assertIsNone(second.last_action_error)

    def test_close_before_completion_is_penalized(self) -> None:
        self.env.step(InvoiceAction(type="list_invoices"))
        result = self.env.step(InvoiceAction(type="close"))

        self.assertTrue(result.done)
        self.assertLess(result.reward, 0.0)
        self.assertIn("closed early", result.message)

    def test_progress_rewards_partial_work(self) -> None:
        for invoice_id, category in [
            ("INV001", "office_supplies"),
            ("INV002", "software"),
            ("INV003", "meals"),
        ]:
            self.env.step(InvoiceAction(type="categorize", invoice_id=invoice_id, category=category))
            result = self.env.step(InvoiceAction(type="validate", invoice_id=invoice_id))

        self.assertAlmostEqual(result.progress, 0.5, places=2)

    def test_state_is_meaningful(self) -> None:
        self.env.step(InvoiceAction(type="view_invoice", invoice_id="INV001"))
        state = self.env.state

        self.assertEqual(state["current_invoice_id"], "INV001")
        self.assertIn("progress", state)
        self.assertIn("total_reward", state)
        self.assertEqual(len(state["invoices"]), 3)

    def test_medium_and_hard_tasks_have_non_approve_outcomes(self) -> None:
        self.env.reset(task="medium")
        medium_outcomes = {invoice["expected_outcome"] for invoice in self.env.invoices.values()}
        self.env.reset(task="hard")
        hard_outcomes = {invoice["expected_outcome"] for invoice in self.env.invoices.values()}

        self.assertIn("reject", medium_outcomes)
        self.assertIn("flag_fraud", hard_outcomes)


if __name__ == "__main__":
    unittest.main()
