import os
from typing import Any, Optional, TYPE_CHECKING

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.containers.runtime import LocalDockerProvider
from websockets.asyncio.client import connect as ws_connect

from models import InvoiceAction, InvoiceObservation

if TYPE_CHECKING:
    from openenv.core.containers.runtime import ContainerProvider


class InvoiceEnv(EnvClient[InvoiceAction, InvoiceObservation, dict]):
    """Typed OpenEnv client for the invoice environment."""

    async def connect(self) -> "InvoiceEnv":
        """Establish a websocket connection with explicit keepalive settings."""
        if self._ws is not None:
            return self

        ws_url_lower = self._ws_url.lower()
        is_localhost = "localhost" in ws_url_lower or "127.0.0.1" in ws_url_lower

        old_no_proxy = os.environ.get("NO_PROXY")
        if is_localhost:
            current_no_proxy = old_no_proxy or ""
            if "localhost" not in current_no_proxy.lower():
                os.environ["NO_PROXY"] = (
                    f"{current_no_proxy},localhost,127.0.0.1"
                    if current_no_proxy
                    else "localhost,127.0.0.1"
                )

        try:
            self._ws = await ws_connect(
                self._ws_url,
                open_timeout=self._connect_timeout,
                max_size=self._max_message_size,
                ping_interval=20,
                ping_timeout=max(20, min(self._message_timeout, 60)),
                close_timeout=5,
            )
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to {self._ws_url}: {exc}") from exc
        finally:
            if is_localhost:
                if old_no_proxy is None:
                    os.environ.pop("NO_PROXY", None)
                else:
                    os.environ["NO_PROXY"] = old_no_proxy

        return self

    @classmethod
    async def from_docker_image_with_timeouts(
        cls,
        image: str,
        *,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 120.0,
        provider: Optional["ContainerProvider"] = None,
        **kwargs: Any,
    ) -> "InvoiceEnv":
        """Start a dockerized env and connect with custom client timeouts."""
        if provider is None:
            provider = LocalDockerProvider()

        base_url = provider.start_container(image, **kwargs)
        provider.wait_for_ready(base_url)

        client = cls(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
        )
        await client.connect()
        return client

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
