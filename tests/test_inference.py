import contextlib
import io
import unittest
from unittest.mock import patch

from inference import InferenceConfig, _run_task_with_retries, format_end_line, format_step_line
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


class InferenceResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_failure_still_emits_structured_start_and_end(self) -> None:
        config = InferenceConfig(
            api_base_url="https://router.huggingface.co/v1",
            model_name="test-model",
            api_key=None,
            task_name="easy",
            benchmark="ap-invoice-env",
            local_image_name="missing-image",
            max_steps=5,
            success_score_threshold=0.65,
            env_connect_timeout_s=1.0,
            env_message_timeout_s=1.0,
            task_retries=0,
        )

        async def failing_run_task(*args, **kwargs):
            raise RuntimeError("boom")

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with patch("inference._run_task", new=failing_run_task):
                await _run_task_with_retries(config, client=None, task_name="easy")

        lines = stdout.getvalue().splitlines()
        self.assertEqual(
            lines[0],
            "[START] task=easy env=ap-invoice-env model=test-model",
        )
        self.assertEqual(
            lines[-1],
            "[END] success=false steps=0 score=0.00 rewards=",
        )


if __name__ == "__main__":
    unittest.main()
