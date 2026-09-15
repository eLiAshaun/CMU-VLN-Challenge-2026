"""Modular client for persistent model workers over abstract Unix sockets.

This module provides a reusable socket_request function for calling
YOLO-World, SAM2, and Qwen3VL persistent servers, without importing the retired monolithic pipeline.
"""

from __future__ import annotations

import json
import time
import uuid
from multiprocessing.connection import Client
from typing import Any


def socket_request(
    endpoint: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    *,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Send a request to a persistent worker over an abstract Unix socket.

    Args:
        endpoint: Abstract Unix socket name (e.g., "@mast3r_yolo")
        payload: Request dict with keys:
            - request_id: str (auto-generated if missing)
            - operation: str (e.g., "healthcheck", "detect_views", "segment_boxes")
            - parameters: dict (operation-specific parameters)
        timeout_seconds: Maximum time to wait for response
        deadline_monotonic: Absolute deadline in time.monotonic() for the request

    Returns:
        Response dict with keys:
            - request_id: str
            - ok: bool
            - metadata: dict (result data)
            - error_code: str (if ok=False)

    Raises:
        ValueError: Invalid endpoint format
        TimeoutError: Response not received within timeout
        RuntimeError: Response request_id mismatch or ok=False
        ConnectionRefusedError: Worker socket not available
    """
    if not endpoint.startswith("@") or len(endpoint) < 2:
        raise ValueError(f"endpoint_must_be_abstract_unix_socket:{endpoint}")

    # Auto-generate request_id if missing
    if "request_id" not in payload:
        payload = {**payload, "request_id": uuid.uuid4().hex}

    # Add deadline to payload if provided
    if deadline_monotonic is not None:
        payload = {**payload, "deadline_monotonic": float(deadline_monotonic)}

    # Calculate effective timeout
    effective_timeout = float(timeout_seconds)
    if deadline_monotonic is not None:
        remaining = max(0.0, deadline_monotonic - time.monotonic())
        effective_timeout = min(effective_timeout, remaining)

    if effective_timeout <= 0.0:
        raise TimeoutError(
            f"model_service_deadline_expired:{payload.get('operation')}:{endpoint}"
        )

    address = "\0" + endpoint[1:]
    connection = Client(address, family="AF_UNIX")
    try:
        request_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        connection.send_bytes(request_bytes)

        if not connection.poll(effective_timeout):
            raise TimeoutError(
                f"model_service_timeout:{payload.get('operation')}:{endpoint}"
            )

        response_bytes = connection.recv_bytes()
        response = json.loads(response_bytes.decode("utf-8"))
    finally:
        connection.close()

    # Validate response
    if response.get("request_id") != payload.get("request_id"):
        raise RuntimeError(
            f"model_service_request_id_mismatch:{payload.get('request_id')} vs {response.get('request_id')}"
        )

    if not response.get("ok"):
        error_code = response.get("error_code", "unknown_error")
        error_detail = response.get("metadata", {}).get("error_detail", "")
        raise RuntimeError(
            f"model_service_error:{error_code}:{error_detail[:200]}"
        )

    return response


def healthcheck(endpoint: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
    """Check if a worker is alive and ready.

    Args:
        endpoint: Abstract Unix socket name
        timeout_seconds: Maximum time to wait

    Returns:
        Worker metadata (device, model_load_seconds, etc.)

    Raises:
        ConnectionRefusedError: Worker not running
        TimeoutError: Worker not responding
    """
    response = socket_request(
        endpoint=endpoint,
        payload={"operation": "healthcheck", "parameters": {}},
        timeout_seconds=timeout_seconds,
    )
    return dict(response.get("metadata", {}))
