import unittest

from inference import format_end_line, format_step_line
from models import InvoiceAction


class InferenceFormattingTests(unittest.TestCase):
    def test_end_line_uses_two_decimal_score(self) -> None:
        line = format_end_line(success=True, steps=3, score=0.6666, rewards=[0.1, 0.2, 0.3])

        self.assertEqual(
            line,
            "[END] success=true steps=3 score=0.67 rewards=0.10,0.20,0.30",
        )

    def test_step_line_reports_error_text(self) -> None:
        line = format_step_line(
            step=2,
            action=InvoiceAction(type="close"),
            reward=-0.25,
            done=True,
            error="Episode closed early with 2 invoices still unresolved.",
        )

        self.assertIn("error=Episode closed early with 2 invoices still unresolved.", line)
        self.assertNotIn("error=null", line)


if __name__ == "__main__":
    unittest.main()
