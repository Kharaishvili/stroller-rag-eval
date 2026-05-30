"""No-op Chroma telemetry implementation for local evaluation runs"""

from __future__ import annotations

from chromadb.config import System
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpTelemetryClient(ProductTelemetryClient):
    """Disable Chroma product telemetry without relying on PostHog internals"""

    def __init__(self, system: System) -> None:
        super().__init__(system)

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return None
