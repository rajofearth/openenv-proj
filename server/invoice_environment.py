from copy import deepcopy
from typing import Any, Dict

from openenv.core.env_server import Environment

from models import InvoiceAction, InvoiceObservation

VALID_CATEGORIES = [
    "office_supplies",
    "software",
    "meals",
    "hardware",
    "logistics",
    "services",
    "marketing",
    "facilities",
]

POLICY_RULES = [
    "Use only these category labels: office_supplies, software, meals, hardware, logistics, services, marketing, facilities.",
    "Invoices above 500 USD require a purchase order before approval.",
    "Future-dated invoices and negative invoice amounts must be rejected.",
    "Unexpected bank-change requests or urgent wire-style wording should be flagged for fraud.",
    "A complete review usually lists the inbox, opens an invoice, categorizes it, validates it, and then finalizes it.",
]


class InvoiceEnv(Environment):
    action_model = InvoiceAction
    observation_model = InvoiceObservation

    def __init__(self) -> None:
        super().__init__()
        self.task_configs = {
            "easy": {
                "objective": "Process three routine invoices with straightforward approvals.",
                "difficulty_notes": "Warm-up task with clear categories and no policy exceptions.",
                "invoices": {
                    "INV001": {
                        "vendor": "OfficeDepot",
                        "amount": 45.99,
                        "date": "2026-04-01",
                        "desc": "Printer paper restock",
                        "ground_category": "office_supplies",
                        "po": None,
                        "currency": "USD",
                        "requester": "Operations",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                    "INV002": {
                        "vendor": "Adobe",
                        "amount": 29.99,
                        "date": "2026-04-02",
                        "desc": "Creative Cloud subscription renewal",
                        "ground_category": "software",
                        "po": None,
                        "currency": "USD",
                        "requester": "Design",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                    "INV003": {
                        "vendor": "Starbucks",
                        "amount": 12.50,
                        "date": "2026-04-03",
                        "desc": "Client coffee meeting",
                        "ground_category": "meals",
                        "po": None,
                        "currency": "USD",
                        "requester": "Sales",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                },
                "max_steps": 15,
            },
            "medium": {
                "objective": "Separate routine approvals from policy rejects in a mixed AP inbox.",
                "difficulty_notes": "Introduces non-fraud policy violations and purchase-order checks.",
                "invoices": {
                    "INV004": {
                        "vendor": "Dell",
                        "amount": 899.00,
                        "date": "2026-04-04",
                        "desc": "Analyst laptop replacement",
                        "ground_category": "hardware",
                        "po": "PO-12345",
                        "currency": "USD",
                        "requester": "IT",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                    "INV005": {
                        "vendor": "Zoom",
                        "amount": 149.99,
                        "date": "2026-04-05",
                        "desc": "Quarterly webinar license",
                        "ground_category": "software",
                        "po": None,
                        "currency": "USD",
                        "requester": "Marketing",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                    "INV006": {
                        "vendor": "FedEx",
                        "amount": 67.50,
                        "date": "2026-04-06",
                        "desc": "Priority shipping for customer samples",
                        "ground_category": "logistics",
                        "po": "PO-67890",
                        "currency": "USD",
                        "requester": "Customer Success",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                    "INV007": {
                        "vendor": "BrightOffice Interiors",
                        "amount": 1200.00,
                        "date": "2026-04-07",
                        "desc": "Standing desks for new hires",
                        "ground_category": "facilities",
                        "po": None,
                        "currency": "USD",
                        "requester": "Workplace",
                        "bank_change_requested": False,
                        "validation_issues": [
                            "missing_purchase_order_for_high_value_invoice"
                        ],
                        "expected_outcome": "reject",
                    },
                },
                "max_steps": 20,
            },
            "hard": {
                "objective": "Handle a realistic mix of approvals, policy rejects, and suspicious invoices.",
                "difficulty_notes": "Requires using policy rules and subtle fraud signals instead of obvious labels.",
                "invoices": {
                    "INV008": {
                        "vendor": "OfficeDepot",
                        "amount": 45.99,
                        "date": "2026-04-01",
                        "desc": "Printer paper restock",
                        "ground_category": "office_supplies",
                        "po": None,
                        "currency": "USD",
                        "requester": "Operations",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                    "INV009": {
                        "vendor": "Meridian Business Services",
                        "amount": 9875.00,
                        "date": "2026-04-08",
                        "desc": "Urgent vendor onboarding payment with updated wire instructions",
                        "ground_category": "services",
                        "po": None,
                        "currency": "USD",
                        "requester": "Finance Ops",
                        "bank_change_requested": True,
                        "validation_issues": [
                            "unexpected_bank_change_request",
                            "high_value_without_purchase_order",
                        ],
                        "expected_outcome": "flag_fraud",
                    },
                    "INV010": {
                        "vendor": "Adobe",
                        "amount": 29.99,
                        "date": "2027-01-01",
                        "desc": "Creative Cloud subscription renewal",
                        "ground_category": "software",
                        "po": None,
                        "currency": "USD",
                        "requester": "Design",
                        "bank_change_requested": False,
                        "validation_issues": ["future_dated_invoice"],
                        "expected_outcome": "reject",
                    },
                    "INV011": {
                        "vendor": "Northwind Hardware",
                        "amount": -150.00,
                        "date": "2026-04-09",
                        "desc": "Adjustment memo submitted as invoice",
                        "ground_category": "hardware",
                        "po": None,
                        "currency": "USD",
                        "requester": "IT",
                        "bank_change_requested": False,
                        "validation_issues": ["negative_invoice_amount"],
                        "expected_outcome": "reject",
                    },
                    "INV012": {
                        "vendor": "Meta",
                        "amount": 450.00,
                        "date": "2026-04-10",
                        "desc": "Paid social campaign spend",
                        "ground_category": "marketing",
                        "po": None,
                        "currency": "USD",
                        "requester": "Growth",
                        "bank_change_requested": False,
                        "validation_issues": [],
                        "expected_outcome": "approve",
                    },
                },
                "max_steps": 25,
            },
        }
        self._load_task("easy")

    def _load_task(self, task: str) -> None:
        cfg = self.task_configs.get(task, self.task_configs["easy"])
        self.current_task = task
        self.task_objective = cfg["objective"]
        self.task_difficulty_notes = cfg["difficulty_notes"]
        self.invoices = deepcopy(cfg["invoices"])
        self.processed: Dict[str, Dict[str, Any]] = {}
        self.current_invoice_id: str | None = None
        self.current_invoice = None
        self.step_count = 0
        self.max_steps = cfg["max_steps"]
        self.total_reward = 0.0
        self.listed_once = False
        self.viewed_invoice_ids: set[str] = set()
        self.last_action_signature: tuple[Any, ...] | None = None
        self.last_action_error: str | None = None

    def _status_for(self, invoice_id: str) -> Dict[str, Any]:
        return self.processed.setdefault(
            invoice_id,
            {
                "category_correct": False,
                "validated": False,
                "resolution": None,
                "correct": None,
                "viewed": False,
            },
        )

    def _is_finalized(self, invoice_id: str) -> bool:
        return self._status_for(invoice_id)["resolution"] is not None

    def _invoice_score(self, invoice_id: str) -> float:
        status = self._status_for(invoice_id)
        score = 0.0
        if status["category_correct"]:
            score += 0.25
        if status["validated"]:
            score += 0.25
        if status["correct"] is True:
            score += 0.50
        return score

    def _progress(self) -> float:
        total = len(self.invoices)
        if total == 0:
            return 0.0
        return (
            sum(self._invoice_score(invoice_id) for invoice_id in self.invoices) / total
        )

    def _finalized_count(self) -> int:
        return sum(1 for invoice_id in self.invoices if self._is_finalized(invoice_id))

    def _correctly_finalized_count(self) -> int:
        return sum(
            1
            for processed in self.processed.values()
            if processed.get("correct") is True
        )

    def _action_signature(self, action: InvoiceAction) -> tuple[Any, ...]:
        return (action.type, action.invoice_id, action.category, action.notes)

    def _public_invoice(
        self, invoice_id: str, invoice: Dict[str, Any]
    ) -> Dict[str, Any]:
        status = self._status_for(invoice_id)
        return {
            "id": invoice_id,
            "vendor": invoice["vendor"],
            "amount": invoice["amount"],
            "date": invoice["date"],
            "desc": invoice["desc"],
            "po": invoice["po"],
            "currency": invoice["currency"],
            "requester": invoice["requester"],
            "bank_change_requested": invoice["bank_change_requested"],
            "processed": self._is_finalized(invoice_id),
            "category_locked_in": status["category_correct"],
            "validated": status["validated"],
            "resolution": status["resolution"],
        }

    def _refresh_current_invoice(self, invoice_id: str | None) -> None:
        if invoice_id is None:
            return
        invoice = self.invoices.get(invoice_id)
        if invoice is None:
            return
        self.current_invoice = self._public_invoice(invoice_id, invoice)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task: str = "easy",
    ) -> InvoiceObservation:
        self._load_task(task)
        return self._build_obs(
            message=f"Inbox loaded with {len(self.invoices)} invoices.",
            progress=0.0,
        )

    def step(
        self,
        action: InvoiceAction,
        timeout_s: float | None = None,
    ) -> InvoiceObservation:
        self.step_count += 1
        reward = 0.0
        message = ""
        error: str | None = None

        signature = self._action_signature(action)
        if signature == self.last_action_signature:
            reward -= 0.03
        self.last_action_signature = signature

        invoice_id = action.invoice_id
        invoice = self.invoices.get(invoice_id) if invoice_id else None

        if invoice_id and invoice is None:
            reward -= 0.10
            error = f"Unknown invoice_id {invoice_id}."
            message = error
        elif (
            action.type in {"categorize", "validate", "approve", "reject", "flag_fraud"}
            and invoice_id
            and self._is_finalized(invoice_id)
        ):
            reward -= 0.10
            error = f"Invoice {invoice_id} is already finalized."
            message = error
        elif action.type == "list_invoices":
            if self.listed_once:
                reward -= 0.02
                error = "Inbox already listed for this episode."
                message = error
            else:
                self.listed_once = True
                reward += 0.05
                message = "Listed all pending invoices."
        elif action.type == "view_invoice" and invoice_id and invoice:
            status = self._status_for(invoice_id)
            self.current_invoice_id = invoice_id
            self.current_invoice = self._public_invoice(invoice_id, invoice)
            if status["viewed"]:
                reward -= 0.02
                error = f"Invoice {invoice_id} was already opened."
                message = error
            else:
                status["viewed"] = True
                self.viewed_invoice_ids.add(invoice_id)
                reward += 0.08
                message = f"Viewed {invoice_id}."
        elif action.type == "categorize" and invoice_id and invoice and action.category:
            status = self._status_for(invoice_id)
            if action.category not in VALID_CATEGORIES:
                reward -= 0.10
                error = f"Unknown category {action.category}. Use one of: {', '.join(VALID_CATEGORIES)}."
                message = error
            elif status["category_correct"]:
                reward -= 0.02
                error = f"Invoice {invoice_id} is already categorized correctly."
                message = error
            elif action.category == invoice["ground_category"]:
                status["category_correct"] = True
                reward += 0.20
                message = f"Correct category for {invoice_id}."
            else:
                reward -= 0.10
                error = f"Wrong category for {invoice_id}."
                message = error
            self._refresh_current_invoice(invoice_id)
        elif action.type == "validate" and invoice_id and invoice:
            status = self._status_for(invoice_id)
            if status["validated"]:
                reward -= 0.02
                error = f"Invoice {invoice_id} has already been validated."
                message = error
            else:
                status["validated"] = True
                if invoice["validation_issues"]:
                    reward += 0.18
                    message = (
                        f"Validation found issues for {invoice_id}: "
                        f"{', '.join(invoice['validation_issues'])}."
                    )
                else:
                    reward += 0.15
                    message = f"Validation passed for {invoice_id}."
            self._refresh_current_invoice(invoice_id)
        elif (
            action.type in {"approve", "reject", "flag_fraud"}
            and invoice_id
            and invoice
        ):
            status = self._status_for(invoice_id)
            if not status["viewed"]:
                reward -= 0.15
                error = f"Invoice {invoice_id} must be opened before final disposition."
                message = error
            elif not status["validated"]:
                reward -= 0.12
                error = (
                    f"Invoice {invoice_id} must be validated before final disposition."
                )
                message = error
            else:
                status["resolution"] = action.type
                status["correct"] = action.type == invoice["expected_outcome"]
                if status["correct"]:
                    reward += 0.35
                    message = f"Correctly finalized {invoice_id} with {action.type}."
                else:
                    reward -= 0.35
                    error = (
                        f"Wrong final disposition for {invoice_id}. Expected "
                        f"{invoice['expected_outcome']}."
                    )
                    message = error
            self._refresh_current_invoice(invoice_id)
        elif action.type == "close":
            remaining = len(self.invoices) - self._finalized_count()
            if remaining > 0:
                reward -= 0.25
                error = (
                    f"Episode closed early with {remaining} invoices still unresolved."
                )
                message = error
            else:
                reward += 0.00
                message = "Episode closed after all invoices were resolved."
        else:
            reward -= 0.08
            error = "Invalid or incomplete action."
            message = error

        self.last_action_error = error
        self.total_reward += reward
        progress = round(self._progress(), 4)
        done = (
            self.step_count >= self.max_steps
            or self._finalized_count() == len(self.invoices)
            or action.type == "close"
        )

        return self._build_obs(
            message=message,
            reward=reward,
            done=done,
            progress=progress,
            last_action_error=error,
        )

    def _build_obs(
        self,
        message: str,
        reward: float = 0.0,
        done: bool = False,
        progress: float = 0.0,
        last_action_error: str | None = None,
    ) -> InvoiceObservation:
        return InvoiceObservation(
            message=message,
            invoices_summary=[
                self._public_invoice(invoice_id, invoice)
                for invoice_id, invoice in self.invoices.items()
            ],
            current_invoice=self.current_invoice,
            valid_categories=VALID_CATEGORIES,
            policy_rules=POLICY_RULES,
            reward=reward,
            done=done,
            progress=progress,
            last_action_error=last_action_error,
            metadata={
                "task": self.current_task,
                "objective": self.task_objective,
                "difficulty_notes": self.task_difficulty_notes,
                "steps": self.step_count,
                "max_steps": self.max_steps,
                "processed": self._finalized_count(),
                "correctly_finalized": self._correctly_finalized_count(),
                "total_invoices": len(self.invoices),
                "total_reward": round(self.total_reward, 2),
            },
        )

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "task": self.current_task,
            "objective": self.task_objective,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_invoice_id": self.current_invoice_id,
            "progress": round(self._progress(), 4),
            "total_reward": round(self.total_reward, 2),
            "listed_once": self.listed_once,
            "last_action_error": self.last_action_error,
            "invoices": [
                self._public_invoice(invoice_id, invoice)
                for invoice_id, invoice in self.invoices.items()
            ],
        }
