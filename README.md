# ap-invoice-env

Real-world Accounts Payable Invoice Processor built as an OpenEnv environment with easy, medium, and hard tasks.

## Quick start

```bash
uv venv
uv pip install -e .[dev]
uv run uvicorn server.app:app --reload --port 8000
```

Create a local env file first:

```bash
cp .example.env .env
```

On Windows PowerShell, if you want to run inference without a dotenv loader, set vars like this:

```powershell
$env:MY_ENV_TASK="easy"
$env:HF_TOKEN="your_hf_token_here"
python inference.py
```

In another terminal:

```bash
curl -X POST http://localhost:8000/reset ^  -H "Content-Type: application/json" ^  -d "{\"task\":\"easy\"}"
```

## Docker

```bash
docker build -f server/Dockerfile -t ap-invoice-env:latest .
```

## Baseline inference

```bash
MY_ENV_TASK=easy python inference.py
MY_ENV_TASK=hard python inference.py
```
