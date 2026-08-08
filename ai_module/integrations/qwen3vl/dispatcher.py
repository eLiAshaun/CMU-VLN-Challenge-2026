from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Mapping

from .model_protocol import ModelRequest, ModelResponse


class LocalModelWorker:
    """Bounded local dispatcher with acquisition cache and strict timeouts."""

    def __init__(self, backends: Mapping[str, object], *, max_cache_entries: int = 128):
        self.backends = dict(backends)
        self.max_cache_entries = max(1, int(max_cache_entries))
        self._cache = OrderedDict()

    def execute(self, request: ModelRequest) -> ModelResponse:
        if request.backend not in self.backends:
            return ModelResponse(request.request_id, False, error_code="backend_unavailable")
        cache_key = hashlib.sha256(repr((request.acquisition_id, request.backend, request.operation, request.input_handles, sorted(request.parameters.items()))).encode()).hexdigest()
        if cache_key in self._cache:
            response = self._cache.pop(cache_key)
            self._cache[cache_key] = response
            return ModelResponse(
                request.request_id,
                response.ok,
                output_handles=response.output_handles,
                metadata=response.metadata,
                error_code=response.error_code,
            )
        try:
            payload = self.backends[request.backend].execute(request)
            response = payload if isinstance(payload, ModelResponse) else ModelResponse(request.request_id, True, metadata=dict(payload))
        except Exception as exc:
            response = ModelResponse(
                request.request_id,
                False,
                metadata={"error_detail": str(exc)[:500]},
                error_code=type(exc).__name__,
            )
        self._cache[cache_key] = response
        while len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        return response
