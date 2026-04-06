from fastapi import FastAPI
import uvicorn
from openenv.core.env_server import create_fastapi_app

from models import InvoiceAction, InvoiceObservation
from server.invoice_environment import InvoiceEnv


def create_app() -> FastAPI:
    return create_fastapi_app(InvoiceEnv, InvoiceAction, InvoiceObservation)


app = create_app()


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
