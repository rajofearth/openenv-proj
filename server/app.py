from openenv.core.env_server import create_fastapi_app

from server.invoice_environment import InvoiceEnv

app = create_fastapi_app(InvoiceEnv)
