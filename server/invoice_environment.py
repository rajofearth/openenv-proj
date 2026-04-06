from copy import deepcopy
from typing import Any, Dict

from openenv.core.env_server import Environment

from models import InvoiceAction, InvoiceObservation


class InvoiceEnv(Environment):
    action_model = InvoiceAction
    observation_model = InvoiceObservation

    def __init__(self) -> None:
        super().__init__()
        self.task_configs = {
            "easy": {
                "invoices": {
                    "INV001": {
                        "vendor": "OfficeDepot",
                        "amount": 45.99,
                        "date": "2026-04-01",
                        "desc": "Paper",
                        "ground_category": "office_supplies",
                        "fraud": False,
                        "po": None,
                    },
                    "INV002": {
                        "vendor": "Adobe",
                        "amount": 29.99,
                        "date": "2026-04-02",
                        "desc": "Subscription",
                        "ground_category": "software",
                        "fraud": False,
                        "po": None,
                    },
                    "INV003": {
                        "vendor": "Starbucks",
                        "amount": 12.50,
                        "date": "2026-04-03",
                        "desc": "Coffee",
                        "ground_category": "meals",
                        "fraud": False,
                        "po": None,
                    },
                },
                "max_steps": 15,
            },
            "medium": {
                "invoices": {
                    "INV004": {
                        "vendor": "Dell",
                        "amount": 899.00,
                        "date": "2026-04-04",
                        "desc": "Laptop",
                        "ground_category": "hardware",
                        "fraud": False,
                        "po": "PO-12345",
                    },
                    "INV005": {
                        "vendor": "Zoom",
                        "amount": 149.99,
                        "date": "2026-04-05",
                        "desc": "Meeting",
                        "ground_category": "software",
                        "fraud": False,
                        "po": None,
                    },
                    "INV006": {
                        "vendor": "FedEx",
                        "amount": 67.50,
                        "date": "2026-04-06",
                        "desc": "Shipping",
                        "ground_category": "logistics",
                        "fraud": False,
                        "po": "PO-67890",
                    },
                    "INV007": {
                        "vendor": "Unknown",
                        "amount": 1200.00,
                        "date": "2026-04-07",
                        "desc": "Consulting",
                        "ground_category": "services",
                        "fraud": False,
                        "po": None,
                    },
                },
                "max_steps": 20,
            },
            "hard": {
                "invoices": {
                    "INV008": {
                        "vendor": "OfficeDepot",
                        "amount": 45.99,
                        "date": "2026-04-01",
                        "desc": "Paper",
                        "ground_category": "office_supplies",
                        "fraud": False,
                        "po": None,
                    },
                    "INV009": {
                        "vendor": "CashApp-Scam",
                        "amount": 9999.00,
                        "date": "2026-04-08",
                        "desc": "Urgent transfer",
                        "ground_category": "other",
                        "fraud": True,
                        "po": None,
                    },
                    "INV010": {
                        "vendor": "Adobe",
                        "amount": 29.99,
                        "date": "2027-01-01",
                        "desc": "Future invoice",
                        "ground_category": "software",
                        "fraud": True,
                        "po": None,
                    },
                    "INV011": {
                        "vendor": "Lenovo",
                        "amount": -150.00,
                        "date": "2026-04-09",
                        "desc": "Refund?",
                        "ground_category": "hardware",
                        "fraud": True,
                        "po": None,
                    },
                    "INV012": {
                        "vendor": "Meta",
                        "amount": 450.00,
                        "date": "2026-04-10",
                        "desc": "Ads",
                        "ground_category": "marketing",
                        "fraud": False,
                        "po": None,
                    },
                },
                "max_steps": 25,
            },
        }
        self._load_task("easy")

    def _load_task(self, task: str) -> None:
        cfg = self.task_configs.get(task, self.task_configs["easy"])
        self.current_task = task
        self.invoices = deepcopy(cfg["invoices"])
        self.processed: Dict[str, Dict[str, Any]] = {}
        self.current_invoice = None
        self.step_count = 0
        self.max_steps = cfg["max_steps"]
        self.last_action_type = None

    async def reset(self, task: str = "easy") -> InvoiceObservation:
        self._load_task(task)
        return self._build_obs(
            f"Inbox loaded with {len(self.invoices)} invoices.",
            progress=0.0,
        )

    async def step(self, action: InvoiceAction) -> InvoiceObservation:
        self.step_count += 1
        reward = 0.0
        message = ""

        if action.type == self.last_action_type:
            reward -= 0.05
        self.last_action_type = action.type

        invoice_id = action.invoice_id
        invoice = self.invoices.get(invoice_id) if invoice_id else None

        if action.type == "list_invoices":
            reward += 0.10
            message = "Listed all pending invoices."
        elif action.type == "view_invoice" and invoice_id and invoice:
            reward += 0.15
            self.current_invoice = {"id": invoice_id, **invoice}
            message = f"Viewed {invoice_id}."
        elif action.type == "categorize" and invoice_id and invoice and action.category:
            if action.category == invoice["ground_category"]:
                reward += 0.35
                self.processed.setdefault(invoice_id, {})["category_correct"] = True
                message = f"Correct category for {invoice_id}."
            else:
                reward -= 0.15
                message = f"Wrong category for {invoice_id}."
        elif action.type == "validate" and invoice_id and invoice:
            valid = invoice["amount"] > 0 and invoice["date"].startswith("2026")
            if valid:
                reward += 0.25
                self.processed.setdefault(invoice_id, {})["validated"] = True
                message = f"Validated {invoice_id}."
            else:
                reward -= 0.10
                message = f"Validation failed for {invoice_id}."
        elif action.type == "approve" and invoice_id and invoice:
            if invoice["fraud"]:
                reward -= 0.40
                self.processed.setdefault(invoice_id, {})["correct"] = False
                message = f"Approved fraudulent invoice {invoice_id}."
            else:
                reward += 0.30
                self.processed.setdefault(invoice_id, {})["correct"] = True
                message = f"Approved {invoice_id}."
        elif action.type == "reject" and invoice_id and invoice:
            if invoice["fraud"]:
                reward += 0.20
                self.processed.setdefault(invoice_id, {})["correct"] = True
                message = f"Rejected suspicious invoice {invoice_id}."
            else:
                reward -= 0.20
                self.processed.setdefault(invoice_id, {})["correct"] = False
                message = f"Rejected non-fraud invoice {invoice_id}."
        elif action.type == "flag_fraud" and invoice_id and invoice:
            if invoice["fraud"]:
                reward += 0.40
                self.processed.setdefault(invoice_id, {})["correct"] = True
                message = f"Correctly flagged fraud {invoice_id}."
            else:
                reward -= 0.20
                self.processed.setdefault(invoice_id, {})["correct"] = False
                message = f"Wrongly flagged legitimate invoice {invoice_id}."
        elif action.type == "close":
            reward += 0.10
            message = "Episode closed."
        else:
            reward -= 0.05
            message = "Invalid or incomplete action."

        correct_count = sum(
            1 for processed in self.processed.values() if processed.get("correct", False)
        )
        total = len(self.invoices)
        progress = correct_count / total if total else 0.0
        done = self.step_count >= self.max_steps or len(self.processed) == total

        return self._build_obs(
            message=message,
            reward=reward,
            done=done,
            progress=progress,
        )

    def _build_obs(
        self,
        message: str,
        reward: float = 0.0,
        done: bool = False,
        progress: float = 0.0,
    ) -> InvoiceObservation:
        return InvoiceObservation(
            message=message,
            invoices_summary=[
                {"id": invoice_id, **invoice, "processed": invoice_id in self.processed}
                for invoice_id, invoice in self.invoices.items()
            ],
            current_invoice=self.current_invoice,
            reward=reward,
            done=done,
            progress=progress,
            metadata={
                "task": self.current_task,
                "steps": self.step_count,
                "processed": len(self.processed),
            },
        )

    def state(self) -> Dict[str, Any]:
        return {"task": self.current_task, "step_count": self.step_count}
