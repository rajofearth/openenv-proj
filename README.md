# ap-invoice-env

Real-world Accounts Payable Invoice Processor built as an OpenEnv environment with easy, medium, and hard tasks.

## Quick start

```bash
uv venv
uv pip install -e .[dev]
uv run uvicorn server.app:app --reload --port 8000
```

In another terminal:

```bash
curl -X POST http://localhost:8000/reset ^
  -H "Content-Type: application/json" ^
  -d "{\"task\":\"easy\"}"
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
