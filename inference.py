from __future__ import annotations

import asyncio
import importlib
import importlib.util
import io
import json
import os
import site
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List


RUNTIME_DEPENDENCIES = ("pydantic", "openai", "websockets", "openenv")
_BOOTSTRAP_ATTEMPTED = False


def _missing_runtime_dependencies() -> list[str]:
    missing: list[str] = []
    for module_name in RUNTIME_DEPENDENCIES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def _ensure_runtime_dependencies() -> None:
    global _BOOTSTRAP_ATTEMPTED

    missing = _missing_runtime_dependencies()
    if not missing:
        return
    if _BOOTSTRAP_ATTEMPTED:
        raise RuntimeError(
            "Missing runtime dependencies after bootstrap attempt: "
            + ", ".join(missing)
        )

    requirements_path = Path(__file__).resolve().with_name("requirements.txt")
    if not requirements_path.exists():
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + f". Bootstrap requires {requirements_path} to exist."
        )

    _BOOTSTRAP_ATTEMPTED = True
    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_path),
    ]

    try:
        subprocess.run(
            install_command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Bootstrap dependency install failed for missing packages "
            + ", ".join(missing)
            + ". Command: "
            + " ".join(install_command)
        ) from exc

    for site_packages_dir in site.getsitepackages():
        site.addsitedir(site_packages_dir)
    user_site_packages = site.getusersitepackages()
    if user_site_packages:
        site.addsitedir(user_site_packages)

    importlib.invalidate_caches()

    remaining = _missing_runtime_dependencies()
    if remaining:
        raise RuntimeError(
            "Runtime dependencies still missing after bootstrap: "
            + ", ".join(remaining)
            + ". Command: "
            + " ".join(install_command)
        )


_ensure_runtime_dependencies()

from models import InvoiceAction, InvoiceObservation

if TYPE_CHECKING:
    from openai import OpenAI
    from client import InvoiceEnv


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
    env_connect_timeout_s: float
    env_message_timeout_s: float
    task_retries: int


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
ALL_TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """
You are an expert accounts payable clerk operating an invoice review environment.

Rules:
- Return exactly one JSON object that matches the action schema.
- Use only the information provided in the observation.
- The only valid categories are the values in `valid_categories`.
- Prefer unfinished invoices and complete a realistic workflow: inspect, categorize, validate, then finalize.
- Use `flag_fraud` only for suspicious payment behavior, not ordinary policy violations.
- Use `reject` for invalid invoices or policy failures.
- Use `approve` only when the invoice is legitimate and passes review.
- Avoid repeating the exact same action unless the environment feedback suggests it.
""".strip()


def _load_config() -> InferenceConfig:
    _load_dotenv()
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
    api_key = (
        os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    )
    task_name = os.getenv("MY_ENV_TASK", "all")
    local_image_name = os.getenv("LOCAL_IMAGE_NAME", "ap-invoice-env:latest")
    env_connect_timeout_s = float(os.getenv("ENV_CONNECT_TIMEOUT_S", "15"))
    env_message_timeout_s = float(os.getenv("ENV_MESSAGE_TIMEOUT_S", "180"))
    task_retries = int(os.getenv("TASK_RETRIES", "2"))

    if not api_key:
        raise RuntimeError(
            "Missing API token. Set HF_TOKEN, OPENAI_API_KEY, or API_KEY before running inference.py."
        )

    if "router.huggingface.co" in api_base_url.lower() and not api_key.startswith(
        "hf_"
    ):
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
        env_connect_timeout_s=env_connect_timeout_s,
        env_message_timeout_s=env_message_timeout_s,
        task_retries=task_retries,
    )


def _single_line(value: Any) -> str:
    return " ".join(str(value).split())


def format_start_line(task: str, env: str, model: str) -> str:
    return f"[START] task={task} env={env} model={model}"


def format_step_line(
    step: int,
    action: InvoiceAction,
    reward: float,
    done: bool,
    error: str | None,
) -> str:
    action_str = _format_action(action)
    error_str = _single_line(error) if error else "null"
    return (
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_str}"
    )


def format_end_line(
    success: bool, steps: int, score: float, rewards: List[float]
) -> str:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    return (
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}"
    )


def _format_action(action: InvoiceAction) -> str:
    if action.type == "categorize":
        return f"categorize({action.invoice_id},{action.category})"
    if action.invoice_id:
        return f"{action.type}({action.invoice_id})"
    return action.type


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


def _build_user_prompt(
    step: int,
    task_name: str,
    observation: InvoiceObservation,
    history: List[str],
) -> str:
    current_invoice = (
        json.dumps(observation.current_invoice)
        if observation.current_invoice
        else "null"
    )
    history_block = "\n".join(history[-8:]) if history else "None"
    return (
        f"Task: {task_name}\n"
        f"Step: {step}\n"
        f"Objective: {observation.metadata.get('objective', '')}\n"
        f"Difficulty notes: {observation.metadata.get('difficulty_notes', '')}\n"
        f"Last environment message: {_single_line(observation.message)}\n"
        f"Last action error: {_single_line(observation.last_action_error or 'null')}\n"
        f"Current invoice: {current_invoice}\n"
        f"Valid categories: {json.dumps(observation.valid_categories)}\n"
        f"Policy rules: {json.dumps(observation.policy_rules)}\n"
        f"Invoices summary: {json.dumps(observation.invoices_summary)}\n"
        f"Progress: {observation.progress:.2f}\n"
        f"Metadata: {json.dumps(observation.metadata)}\n"
        f"Recent history:\n{history_block}\n"
        "Return exactly one action as a JSON object."
    )


def _build_completion_kwargs(
    config: InferenceConfig,
    step: int,
    task_name: str,
    observation: InvoiceObservation,
    history: List[str],
    include_response_format: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    step=step,
                    task_name=task_name,
                    observation=observation,
                    history=history,
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 180,
    }
    if include_response_format:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": ACTION_JSON_SCHEMA,
        }
    return payload


def _fallback_action(observation: InvoiceObservation) -> InvoiceAction:
    if observation.current_invoice and not observation.current_invoice.get("processed"):
        return InvoiceAction(
            type="validate", invoice_id=observation.current_invoice["id"]
        )
    return InvoiceAction(type="list_invoices")


def _get_model_action(
    client: Any,
    config: InferenceConfig,
    step: int,
    task_name: str,
    observation: InvoiceObservation,
    history: List[str],
) -> InvoiceAction:
    errors: List[str] = []
    for include_response_format in (True, False):
        try:
            completion = client.chat.completions.create(
                **_build_completion_kwargs(
                    config=config,
                    step=step,
                    task_name=task_name,
                    observation=observation,
                    history=history,
                    include_response_format=include_response_format,
                )
            )
            raw_text = _extract_message_text(completion.choices[0].message)
            action_dict = _extract_json_object(raw_text)
            return InvoiceAction.model_validate(action_dict)
        except Exception as exc:
            errors.append(_single_line(exc))

    history.append("model_error=" + " | ".join(errors))
    return _fallback_action(observation)


def _task_list(task_name: str) -> List[str]:
    requested = task_name.strip().lower()
    if requested in {"", "all", "*"}:
        return ALL_TASKS
    return [task_name]


async def _run_task(config: InferenceConfig, client: Any, task_name: str) -> None:
    env = await _open_env(config)
    rewards: List[float] = []
    history: List[str] = []
    action_history: List[InvoiceAction] = []
    steps_taken = 0
    score = 0.0
    success = False

    print(
        format_start_line(
            task=task_name, env=config.benchmark, model=config.model_name
        ),
        flush=True,
    )

    try:
        result = await env.reset(task=task_name)
        observation = result.observation

        for step in range(1, config.max_steps + 1):
            if result.done:
                break

            action = _get_model_action(
                client=client,
                config=config,
                step=step,
                task_name=task_name,
                observation=observation,
                history=history,
            )
            result, env = await _step_with_recovery(
                config=config,
                env=env,
                task_name=task_name,
                action_history=action_history,
                action=action,
            )
            observation = result.observation
            reward = result.reward or 0.0
            rewards.append(reward)
            action_history.append(action)
            steps_taken = step

            print(
                format_step_line(
                    step=step,
                    action=action,
                    reward=reward,
                    done=result.done,
                    error=observation.last_action_error,
                ),
                flush=True,
            )

            history.append(
                _single_line(
                    f"step={step} action={action.model_dump(exclude_none=True)} "
                    f"reward={reward:.2f} done={str(result.done).lower()} "
                    f"progress={observation.progress:.2f} "
                    f"error={observation.last_action_error or 'null'}"
                )
            )

            if result.done:
                break

        score = min(max(observation.progress, 0.0), 1.0)
        success = score >= config.success_score_threshold
    finally:
        try:
            await env.close()
        except Exception:
            pass
        print(
            format_end_line(
                success=success, steps=steps_taken, score=score, rewards=rewards
            ),
            flush=True,
        )


async def _open_env(config: InferenceConfig) -> InvoiceEnv:
    from client import InvoiceEnv

    return await InvoiceEnv.from_docker_image_with_timeouts(
        config.local_image_name,
        connect_timeout_s=config.env_connect_timeout_s,
        message_timeout_s=config.env_message_timeout_s,
    )


async def _rebuild_env(
    config: InferenceConfig,
    task_name: str,
    action_history: List[InvoiceAction],
) -> InvoiceEnv:
    env = await _open_env(config)
    await env.reset(task=task_name)
    for prior_action in action_history:
        await env.step(prior_action)
    return env


async def _step_with_recovery(
    config: InferenceConfig,
    env: InvoiceEnv,
    task_name: str,
    action_history: List[InvoiceAction],
    action: InvoiceAction,
) -> tuple[Any, InvoiceEnv]:
    attempts = max(1, config.task_retries + 1)
    current_env = env

    for attempt in range(1, attempts + 1):
        try:
            return await current_env.step(action), current_env
        except Exception as exc:
            if not _is_transient_env_error(exc) or attempt >= attempts:
                raise

            print(
                f"Recovering task {task_name}: reconnecting after websocket error while handling "
                f"{_format_action(action)} ({attempt}/{attempts}). Reason: {_single_line(exc)}",
                file=sys.stderr,
                flush=True,
            )

            try:
                await current_env.close()
            except Exception:
                pass

            await asyncio.sleep(min(attempt, 3))
            current_env = await _rebuild_env(
                config=config,
                task_name=task_name,
                action_history=action_history,
            )

    raise RuntimeError("Unreachable recovery flow")


def _is_transient_env_error(exc: Exception) -> bool:
    websocket_error_types: tuple[type[BaseException], ...] = ()
    try:
        from websockets import exceptions as websocket_exceptions

        websocket_error_types = (websocket_exceptions.WebSocketException,)
    except Exception:
        pass

    return isinstance(
        exc,
        (
            asyncio.TimeoutError,
            ConnectionError,
            ConnectionAbortedError,
            OSError,
            *websocket_error_types,
        ),
    )


async def _run_task_with_retries(
    config: InferenceConfig, client: Any, task_name: str
) -> None:
    attempts = max(1, config.task_retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            await _run_task(config, client, task_name)
            return
        except Exception as exc:
            if not _is_transient_env_error(exc) or attempt >= attempts:
                raise
            print(
                f"Retrying task {task_name} from recovered state ({attempt}/{attempts}). "
                f"Reason: {_single_line(exc)}",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.sleep(min(attempt, 3))


async def main() -> None:
    from openai import OpenAI

    config = _load_config()
    client = OpenAI(base_url=config.api_base_url, api_key=config.api_key)
    for task_name in _task_list(config.task_name):
        await _run_task_with_retries(config, client, task_name)


if __name__ == "__main__":
    original_stdout = sys.stdout
    stdout_buffer = io.StringIO()
    sys.stdout = stdout_buffer
    exit_code = 0
    try:
        asyncio.run(main())
    except Exception:
        exit_code = 1
    finally:
        sys.stdout = original_stdout

    for line in stdout_buffer.getvalue().splitlines():
        if line.startswith(("[START]", "[STEP]", "[END]")):
            print(line, flush=True)

    raise SystemExit(exit_code)
