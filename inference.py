import asyncio
import json
import os
from typing import Any, List

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
TASK_NAME = os.getenv("MY_ENV_TASK", "all")
BENCHMARK = "ap-invoice-env"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "ap-invoice-env:latest")
MAX_STEPS = 25
SUCCESS_SCORE_THRESHOLD = 0.5
ALL_TASKS = ["easy", "medium", "hard"]
ACTION_JSON_SCHEMA = {
    "name": "invoice_action",
    "schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": [
                    "list_invoices",
                    "view_invoice",
                    "categorize",
                    "validate",
                    "approve",
                    "reject",
                    "flag_fraud",
                    "close",
                ],
            },
            "invoice_id": {"type": ["string", "null"]},
            "category": {"type": ["string", "null"]},
            "notes": {"type": ["string", "null"]},
        },
        "required": ["type"],
        "additionalProperties": False,
    },
}


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
You are an expert accounts payable clerk.
Your job is to process invoices correctly using only the available actions.

Rules:
- Return exactly one JSON object.
- Prefer deliberate progress over repeating the same action.
- Use the invoice data in the observation only.
- Do not act on invoices where `processed` is already true.
- Prefer unfinished invoices and move the workflow forward.
- Legitimate invoices should usually be categorized, validated, and approved.
- Suspicious invoices should be flagged for fraud or rejected when appropriate.
- Close only when the work is complete or no further progress is possible.
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


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No JSON object found in model response.")


def _build_user_prompt(step: int, task_name: str, last_obs: Any, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    current_invoice = json.dumps(last_obs.current_invoice) if last_obs.current_invoice else "null"
    return (
        f"Task: {task_name}\n"
        f"Step: {step}\n"
        f"Last environment message: {last_obs.message}\n"
        f"Current invoice: {current_invoice}\n"
        f"Invoices summary: {json.dumps(last_obs.invoices_summary)}\n"
        f"Progress: {last_obs.progress:.2f}\n"
        f"Metadata: {json.dumps(last_obs.metadata)}\n"
        f"Recent history:\n{history_block}\n"
        "Important: invoices with processed=true are already finalized. "
        "Choose an unfinished invoice unless everything is complete.\n"
        "Return exactly one action as a JSON object that matches the schema."
    )


def _build_completion_kwargs(
    step: int, task_name: str, last_obs: Any, history: List[str], include_response_format: bool
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(step, task_name, last_obs, history)},
        ],
        "temperature": 0.2,
        "max_tokens": 200,
    }
    if include_response_format:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": ACTION_JSON_SCHEMA,
        }
    return payload


def _get_model_action(step: int, task_name: str, last_obs: Any, history: List[str]) -> InvoiceAction:
    for include_response_format in (True, False):
        try:
            completion = client.chat.completions.create(
                **_build_completion_kwargs(
                    step=step,
                    task_name=task_name,
                    last_obs=last_obs,
                    history=history,
                    include_response_format=include_response_format,
                )
            )
            raw_text = _extract_message_text(completion.choices[0].message)
            action_dict = _extract_json_object(raw_text)
            return InvoiceAction.model_validate(action_dict)
        except Exception:
            continue
    return InvoiceAction(type="list_invoices")


def _task_list() -> List[str]:
    requested = TASK_NAME.strip().lower()
    if requested in {"", "all", "*"}:
        return ALL_TASKS
    return [TASK_NAME]


async def _run_task(task_name: str) -> None:
    env = await InvoiceEnv.from_docker_image(LOCAL_IMAGE_NAME)
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        result = await env.reset(task=task_name)
        last_obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = _get_model_action(step, task_name, last_obs, history)
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} "
                f"done={str(result.done).lower()} error=null",
                flush=True,
            )

            history.append(
                f"step={step} action={action_str} reward={reward:.2f} "
                f"done={str(result.done).lower()} progress={obs.progress:.2f}"
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


async def main() -> None:
    for task_name in _task_list():
        await _run_task(task_name)


if __name__ == "__main__":
    asyncio.run(main())
