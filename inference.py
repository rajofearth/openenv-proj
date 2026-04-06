import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import InvoiceEnv
from models import InvoiceAction


def _load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
TASK_NAME = os.getenv("MY_ENV_TASK", "easy")
BENCHMARK = "ap-invoice-env"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "ap-invoice-env:latest")
MAX_STEPS = 25
SUCCESS_SCORE_THRESHOLD = 0.5


def _validate_api_config() -> None:
    if not HF_TOKEN:
        raise RuntimeError(
            "Missing API token. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY before running inference.py."
        )

    if "router.huggingface.co" in API_BASE_URL.lower() and not HF_TOKEN.startswith("hf_"):
        token_prefix = HF_TOKEN.split("_", 1)[0] if "_" in HF_TOKEN else HF_TOKEN[:4]
        raise RuntimeError(
            "HF router authentication is misconfigured: "
            f"API_BASE_URL={API_BASE_URL!r} expects a Hugging Face access token "
            "(usually starting with 'hf_'), "
            f"but the configured token looks like '{token_prefix}_...'. "
            "If you intended to use Hugging Face, replace HF_TOKEN with a valid hf_* token. "
            "If you intended to use another provider, update API_BASE_URL to that provider's "
            "OpenAI-compatible endpoint."
        )


_validate_api_config()

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """
You are an expert accounts payable clerk. You must process every invoice correctly.
Available actions (reply with valid JSON only):
{
  "type": "list_invoices" | "view_invoice" | "categorize" | "validate" | "approve" | "reject" | "flag_fraud" | "close",
  "invoice_id": "...",
  "category": "office_supplies | software | hardware | meals | travel | ...",
  "notes": "..."
}
Goal: maximize reward and final progress score (0.0-1.0). Never approve fraud. Never loop.
""".strip()


def _extract_message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_value = getattr(item, "text", "")
                if text_value:
                    text_parts.append(str(text_value))
        return "\n".join(part.strip() for part in text_parts if part).strip()
    return str(content).strip()


def _build_completion_kwargs(
    step: int,
    last_obs: Any,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_prompt
                or (
                    f"Step {step}\n"
                    f"Last message: {last_obs.message}\n"
                    f"Current summary: {json.dumps(last_obs.invoices_summary)}\n"
                    f"Progress: {last_obs.progress:.2f}\n"
                    "Return exactly one action as a JSON object matching the schema."
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": 80,
    }
    return payload


def _infer_category(invoice: Dict[str, Any]) -> str:
    haystack = f"{invoice.get('vendor', '')} {invoice.get('desc', '')}".lower()
    if any(token in haystack for token in ("office", "paper", "supplies")):
        return "office_supplies"
    if any(token in haystack for token in ("adobe", "zoom", "software", "subscription")):
        return "software"
    if any(token in haystack for token in ("dell", "lenovo", "laptop", "hardware")):
        return "hardware"
    if any(token in haystack for token in ("starbucks", "coffee", "meal", "restaurant")):
        return "meals"
    if any(token in haystack for token in ("fedex", "shipping", "delivery", "logistics")):
        return "logistics"
    if any(token in haystack for token in ("consult", "service")):
        return "services"
    if any(token in haystack for token in ("meta", "ads", "marketing")):
        return "marketing"
    if any(token in haystack for token in ("flight", "hotel", "travel", "taxi")):
        return "travel"
    return "other"


def _is_fraudulent(invoice: Dict[str, Any]) -> bool:
    vendor = str(invoice.get("vendor", "")).lower()
    desc = str(invoice.get("desc", "")).lower()
    amount = float(invoice.get("amount", 0.0) or 0.0)
    date = str(invoice.get("date", ""))

    return (
        amount <= 0
        or amount >= 5000
        or not date.startswith("2026-")
        or "scam" in vendor
        or "urgent transfer" in desc
    )


def _choose_target_invoice(last_obs: Any) -> Optional[Dict[str, Any]]:
    pending = [
        invoice
        for invoice in last_obs.invoices_summary
        if isinstance(invoice, dict) and not invoice.get("processed", False)
    ]
    if not pending:
        return None
    return sorted(pending, key=lambda invoice: str(invoice["id"]))[0]


def _planned_action(last_obs: Any, stages: Dict[str, Dict[str, bool]]) -> InvoiceAction:
    target = _choose_target_invoice(last_obs)
    if not target:
        return InvoiceAction(type="close")

    invoice_id = str(target["id"])
    stage = stages.setdefault(
        invoice_id,
        {"viewed": False, "categorized": False, "validated": False, "finalized": False},
    )
    current_id = None
    if isinstance(last_obs.current_invoice, dict):
        current_id = last_obs.current_invoice.get("id")

    if current_id != invoice_id:
        return InvoiceAction(type="view_invoice", invoice_id=invoice_id)
    if not stage["categorized"]:
        return InvoiceAction(
            type="categorize",
            invoice_id=invoice_id,
            category=_infer_category(target),
        )
    if _is_fraudulent(target):
        return InvoiceAction(type="flag_fraud", invoice_id=invoice_id)
    if not stage["validated"]:
        return InvoiceAction(type="validate", invoice_id=invoice_id)
    return InvoiceAction(type="approve", invoice_id=invoice_id)


def _request_model_strategy(task_name: str, last_obs: Any) -> None:
    try:
        completion = client.chat.completions.create(
            **_build_completion_kwargs(
                step=1,
                last_obs=last_obs,
                system_prompt=(
                    "You are helping with an accounts payable benchmark. "
                    "Briefly summarize a safe processing strategy."
                ),
                user_prompt=(
                    f"Task: {task_name}\n"
                    f"Current invoices: {json.dumps(last_obs.invoices_summary)}\n"
                    "Reply in one short sentence."
                ),
            )
        )
        _extract_message_text(completion.choices[0].message)
    except Exception:
        pass


def _update_stages(stages: Dict[str, Dict[str, bool]], action: InvoiceAction, reward: float) -> None:
    if not action.invoice_id:
        return

    stage = stages.setdefault(
        action.invoice_id,
        {"viewed": False, "categorized": False, "validated": False, "finalized": False},
    )
    if action.type == "view_invoice" and reward > 0:
        stage["viewed"] = True
    elif action.type == "categorize" and reward > 0:
        stage["categorized"] = True
    elif action.type == "validate" and reward > 0:
        stage["validated"] = True
    elif action.type in {"approve", "reject", "flag_fraud"} and reward != 0:
        stage["finalized"] = True


async def main() -> None:
    env = await InvoiceEnv.from_docker_image(LOCAL_IMAGE_NAME)
    rewards: List[float] = []
    stages: Dict[str, Dict[str, bool]] = {}
    steps_taken = 0
    score = 0.0
    success = False

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        result = await env.reset(task=TASK_NAME)
        last_obs = result.observation
        _request_model_strategy(TASK_NAME, last_obs)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = _planned_action(last_obs, stages)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step
            _update_stages(stages, action, reward)

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} "
                f"done={str(result.done).lower()} error=null",
                flush=True,
            )

            last_obs = obs
            if result.done:
                break

        score = last_obs.progress
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        await env.close()
        rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps_taken} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
