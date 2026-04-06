from openenv.core.env_server import create_fastapi_app

from server.invoice_environment import InvoiceEnv
from models import InvoiceAction, InvoiceObservation

app = create_fastapi_app(InvoiceEnv, InvoiceAction, InvoiceObservation)
