from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class InvoiceAction(BaseModel):
    type: Literal[
        "list_invoices",
        "view_invoice",
        "categorize",
        "validate",
        "approve",
        "reject",
        "flag_fraud",
        "close",
    ]
    invoice_id: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


class InvoiceObservation(BaseModel):
    message: str
    invoices_summary: List[Dict[str, Any]] = Field(default_factory=list)
    current_invoice: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    done: bool = False
    progress: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
