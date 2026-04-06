from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import InvoiceAction, InvoiceObservation


class InvoiceEnv(EnvClient[InvoiceAction, InvoiceObservation, dict]):
    """Typed OpenEnv client for the invoice environment."""

    def _step_payload(self, action: InvoiceAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[InvoiceObservation]:
        observation = InvoiceObservation.model_validate(
            {
                **payload.get("observation", {}),
                "reward": payload.get("reward") or 0.0,
                "done": payload.get("done", False),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> dict:
        return payload
