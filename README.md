# ap-invoice-env

Real-world Accounts Payable Invoice Processor built as an OpenEnv environment with easy, medium, and hard tasks.
MAKE SURE IT FOLLOWS: [READTHIS](READTHIS.md)

## Overview

This environment simulates a small accounts payable inbox. An agent must inspect invoices, assign a category, validate basic business fields, detect suspicious submissions, and make the correct final disposition.

Why this environment is useful:

- It models a real back-office workflow instead of a toy game.
- It provides shaped rewards for partial progress while keeping final scoring tied to correct outcomes.
- It includes increasing difficulty from routine invoices to adversarial fraud-like edge cases.

## Action Space

The agent sends a typed `InvoiceAction` with:

- `type`: `list_invoices`, `view_invoice`, `categorize`, `validate`, `approve`, `reject`, `flag_fraud`, or `close`
- `invoice_id`: optional invoice identifier for invoice-specific actions
- `category`: optional category label for `categorize`
- `notes`: optional notes field

## Observation Space

Each step returns a typed `InvoiceObservation` with:

- `message`: feedback for the last action
- `invoices_summary`: invoice list with business-relevant fields and completion status
- `current_invoice`: the currently opened invoice, if any
- `reward`: shaped step reward
- `done`: whether the episode has ended
- `progress`: normalized score in `[0, 1]`
- `metadata`: task name, step count, and finalized invoice count

## Tasks

- `easy`: 3 routine legitimate invoices across office supplies, software, and meals
- `medium`: 4 legitimate invoices with broader category coverage and higher-value purchases
- `hard`: 5 invoices mixing legitimate work with suspicious vendors, impossible dates, and abnormal amounts

## Quick start

```bash
uv venv
uv pip install -e .[dev]
uv run uvicorn server.app:app --reload --port 8000
```

Create a local env file first:

```powershell
Copy-Item .env.example .env
```

`inference.py` auto-loads values from `.env` if that file exists.

On Windows PowerShell, you can also set vars directly:

```powershell
$env:MY_ENV_TASK="easy"
$env:HF_TOKEN="hf_your_huggingface_token_here"
python inference.py
```

When `API_BASE_URL` is `https://router.huggingface.co/v1`, `HF_TOKEN` must be a Hugging Face access token. Tokens from other providers will be rejected.

In another terminal:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/reset" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"task":"easy"}'
```

## Docker

```bash
docker build -f server/Dockerfile -t ap-invoice-env:latest .
```

## Validation

Run the OpenEnv validator before submitting:

```bash
uv run openenv validate
```

## Baseline inference

The baseline script uses the OpenAI client to let the model choose actions from the current observation. This keeps the project a benchmark for agent decision-making rather than a hand-scripted solver.

```powershell
docker build -f server/Dockerfile -t ap-invoice-env:latest .
python inference.py
$env:MY_ENV_TASK="easy"; python inference.py
$env:MY_ENV_TASK="medium"; python inference.py
$env:MY_ENV_TASK="hard"; python inference.py
```

If `MY_ENV_TASK` is unset or set to `all`, the script runs `easy`, `medium`, and `hard` sequentially.
