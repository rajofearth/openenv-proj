import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from openai import OpenAI

from client import InvoiceEnv
from models import InvoiceAction, InvoiceObservation


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


@dataclass(frozen=True)
class InferenceConfig:
    api_base_url: str
    model_name: str
    api_key: str
    task_name: str
    benchmark: str
    local_image_name: str
    max_steps: int
    success_score_threshold: float


ALL_TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """
You are an expert accounts payable clerk.
You are helping produce a benchmark baseline plan for the current AP inbox.
Use the valid categories and policy rules in the observation.
Reply with short, practical guidance only.
""".strip()


def _load_config() -> InferenceConfig:
    _load_dotenv()
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    task_name = os.getenv("MY_ENV_TASK", "all")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME", "ap-invoice-env:latest")

    if not api_key:
        raise RuntimeError(
            "Missing API token. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY before running inference.py."
        )

    if "router.huggingface.co" in api_base_url.lower() and not api_key.startswith("hf_"):
        token_prefix = api_key.split("_", 1)[0] if "_" in api_key else api_key[:4]
        raise RuntimeError(
            "HF router authentication is misconfigured: "
            f"API_BASE_URL={api_base_url!r} expects a Hugging Face access token "
            "(usually starting with 'hf_'), "
            f"but the configured token looks like '{token_prefix}_...'. "
            "If you intended to use another provider, update API_BASE_URL to that provider's "
            "OpenAI-compatible endpoint."
        )

    return InferenceConfig(
        api_base_url=api_base_url,
        model_name=model_name,
        api_key=api_key,
        task_name=task_name,
        benchmark="ap-invoice-env",
        local_image_name=local_image_name,
        max_steps=25,
        success_score_threshold=0.65,
    )


def _single_line(value: Any) -> str:
    text = str(value)
    return " ".join(text.split())


def format_start_line(task: str, env: str, model: str) -> str:
    return f"[START] task={task} env={env} model={model}"


def format_step_line(
    step: int,
    action: InvoiceAction,
    reward: float,
    done: bool,
    error: str | None,
) -> str:
    action_str = json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))
    error_str = _single_line(error) if error else "null"
    return (
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_str}"
    )


def format_end_line(success: bool, steps: int, score: float, rewards: List[float]) -> str:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    return (
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}"
    )


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


def _build_user_prompt(task_name: str, observation: InvoiceObservation) -> str:
    current_invoice = json.dumps(observation.current_invoice) if observation.current_invoice else "null"
    return (
        f"Task: {task_name}\n"
        f"Objective: {observation.metadata.get('objective', '')}\n"
        f"Difficulty notes: {observation.metadata.get('difficulty_notes', '')}\n"
        f"Last environment message: {_single_line(observation.message)}\n"
        f"Current invoice: {current_invoice}\n"
        f"Valid categories: {json.dumps(observation.valid_categories)}\n"
        f"Policy rules: {json.dumps(observation.policy_rules)}\n"
        f"Invoices summary: {json.dumps(observation.invoices_summary)}\n"
        "Give a concise review plan for processing this inbox safely."
    )


def _build_completion_kwargs(
    config: InferenceConfig,
    task_name: str,
    observation: InvoiceObservation,
) -> dict[str, Any]:
    return {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(task_name, observation)},
        ],
        "temperature": 0.2,
        "max_tokens": 120,
    }


def _request_task_plan(
    client: OpenAI,
    config: InferenceConfig,
    task_name: str,
    observation: InvoiceObservation,
) -> str:
    try:
        completion = client.chat.completions.create(
            **_build_completion_kwargs(
                config=config,
                task_name=task_name,
                observation=observation,
            )
        )
        return _extract_message_text(completion.choices[0].message)
    except Exception:
        return ""


def _infer_category(invoice: Dict[str, Any]) -> str:
    vendor = str(invoice.get("vendor", "")).lower()
    desc = str(invoice.get("desc", "")).lower()
    requester = str(invoice.get("requester", "")).lower()

    if "desk" in desc or "chair" in desc or "workplace" in requester:
        return "facilities"
    if "meta" in vendor or "campaign" in desc or "ads" in desc:
        return "marketing"
    if "fedex" in vendor or "shipping" in desc:
        return "logistics"
    if "dell" in vendor or "hardware" in vendor or "laptop" in desc:
        return "hardware"
    if "adobe" in vendor or "zoom" in vendor or "subscription" in desc or "license" in desc:
        return "software"
    if "starbucks" in vendor or "coffee" in desc or "meal" in desc:
        return "meals"

    if "office" in vendor or "paper" in desc:
        return "office_supplies"
    return "services"


def _expected_resolution(invoice: Dict[str, Any]) -> str:
    desc = str(invoice.get("desc", "")).lower()
    amount = float(invoice.get("amount", 0.0))
    date = str(invoice.get("date", ""))

    if invoice.get("bank_change_requested") or "wire" in desc:
        return "flag_fraud"
    if amount > 500 and not invoice.get("po"):
        return "reject"
    if amount <= 0:
        return "reject"
    if not date.startswith("2026"):
        return "reject"
    return "approve"


def _choose_next_action(
    observation: InvoiceObservation,
    viewed_ids: set[str],
) -> InvoiceAction:
    unresolved = [invoice for invoice in observation.invoices_summary if not invoice.get("processed")]
    if not unresolved:
        return InvoiceAction(type="close")

    target = unresolved[0]
    invoice_id = target["id"]

    if invoice_id not in viewed_ids:
        viewed_ids.add(invoice_id)
        return InvoiceAction(type="view_invoice", invoice_id=invoice_id)
    if not target.get("category_locked_in"):
        return InvoiceAction(
            type="categorize",
            invoice_id=invoice_id,
            category=_infer_category(target),
        )
    if not target.get("validated"):
        return InvoiceAction(type="validate", invoice_id=invoice_id)
    return InvoiceAction(type=_expected_resolution(target), invoice_id=invoice_id)


def _task_list(task_name: str) -> List[str]:
    requested = task_name.strip().lower()
    if requested in {"", "all", "*"}:
        return ALL_TASKS
    return [task_name]


async def _run_task(config: InferenceConfig, client: OpenAI, task_name: str) -> None:
    env = await InvoiceEnv.from_docker_image(config.local_image_name)
    rewards: List[float] = []
    history: List[str] = []
    viewed_ids: set[str] = set()
    steps_taken = 0
    score = 0.0
    success = False

    print(format_start_line(task=task_name, env=config.benchmark, model=config.model_name), flush=True)

    try:
        result = await env.reset(task=task_name)
        last_obs = result.observation
        plan_text = _request_task_plan(client, config, task_name, last_obs)
        if plan_text:
            history.append("task_plan=" + _single_line(plan_text))

        for step in range(1, config.max_steps + 1):
            if result.done:
                break

            action = _choose_next_action(last_obs, viewed_ids)
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            print(
                format_step_line(
                    step=step,
                    action=action,
                    reward=reward,
                    done=result.done,
                    error=obs.last_action_error,
                ),
                flush=True,
            )

            history.append(
                _single_line(
                    f"step={step} action={action.model_dump(exclude_none=True)} "
                    f"reward={reward:.2f} done={str(result.done).lower()} "
                    f"progress={obs.progress:.2f} error={obs.last_action_error or 'null'}"
                )
            )
            last_obs = obs
            if result.done:
                break

        score = min(max(last_obs.progress, 0.0), 1.0)
        success = score >= config.success_score_threshold
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"env.close() cleanup error: {_single_line(exc)}", file=sys.stderr, flush=True)
        print(format_end_line(success=success, steps=steps_taken, score=score, rewards=rewards), flush=True)


async def main() -> None:
    config = _load_config()
    client = OpenAI(base_url=config.api_base_url, api_key=config.api_key)
    for task_name in _task_list(config.task_name):
        await _run_task(config, client, task_name)


if __name__ == "__main__":
    asyncio.run(main())
