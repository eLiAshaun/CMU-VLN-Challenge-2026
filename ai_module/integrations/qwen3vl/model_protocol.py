from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class ModelRequest:
    request_id: str
    acquisition_id: str
    backend: str
    operation: str
    input_handles: tuple[str, ...] = ()
    parameters: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    request_id: str
    ok: bool
    output_handles: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    error_code: str = ""
