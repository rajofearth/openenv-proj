from openenv.core.http_client import HTTPEnvClient

from models import InvoiceAction, InvoiceObservation


class InvoiceEnv(HTTPEnvClient[InvoiceAction, InvoiceObservation]):
    """OpenEnv HTTP client compatible with from_docker_image()."""

    pass
