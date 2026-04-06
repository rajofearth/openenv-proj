import asyncio
import json
import os
from typing import List

from openai import OpenAI

from client import InvoiceEnv
from models import InvoiceAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("MY_ENV_TASK", "easy")
BENCHMARK = "ap-invoice-env"
MAX_STEPS = 25
SUCCESS_SCORE_THRESHOLD = 0.5

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


async def main() -> None:
    env = await InvoiceEnv.from_docker_image("ap-invoice-env:latest")
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        result = await env.reset(task=TASK_NAME)
        last_obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if last_obs.done:
                break

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Step {step}\n"
                            f"Last message: {last_obs.message}\n"
                            f"Current summary: {json.dumps(last_obs.invoices_summary)}\n"
                            f"Progress: {last_obs.progress:.2f}\n"
                            "Reply ONLY with JSON action."
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            try:
                action_dict = json.loads(completion.choices[0].message.content)
                action = InvoiceAction.model_validate(action_dict)
            except Exception:
                action = InvoiceAction(type="list_invoices")

            result = await env.step(action)
            obs = result.observation
            reward = obs.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            print(
                f"[STEP] step={step} action={action_str} reward={reward:.2f} "
                f"done={str(obs.done).lower()} error=null",
                flush=True,
            )

            last_obs = obs
            if obs.done:
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
