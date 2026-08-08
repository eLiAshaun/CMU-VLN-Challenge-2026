#!/usr/bin/env python3
"""Abstract-Unix-socket wrapper around the local Qwen3-VL worker."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener

from .backend import Qwen3VLBackend
from .dispatcher import LocalModelWorker
from .local import LocalQwen3VLImplementation
from .model_protocol import ModelRequest


def _request_from_json(payload) -> ModelRequest:
    if not isinstance(payload, dict):
        raise ValueError("request_must_be_object")
    return ModelRequest(
        request_id=str(payload["request_id"]),
        acquisition_id=str(payload["acquisition_id"]),
        backend=str(payload["backend"]),
        operation=str(payload["operation"]),
        input_handles=tuple(
            str(value) for value in payload.get("input_handles", ())
        ),
        parameters=dict(payload.get("parameters", {})),
    )


def serve(
    checkpoint: str,
    endpoint: str,
    quantization: str = "int8",
    max_pixels: int = 512 * 512,
) -> None:
    if not endpoint.startswith("@") or len(endpoint) < 2:
        raise ValueError("endpoint_must_be_abstract_unix_socket")
    address = "\0" + endpoint[1:]
    worker = LocalModelWorker({
        "qwen3vl": Qwen3VLBackend(
            LocalQwen3VLImplementation(
                checkpoint,
                quantization=quantization,
                max_pixels=max_pixels,
            )
        ),
    })
    listener = Listener(address, family="AF_UNIX")
    try:
        while True:
            connection = listener.accept()
            try:
                payload = json.loads(
                    connection.recv_bytes().decode("utf-8")
                )
                request = _request_from_json(payload)
                response = worker.execute(request)
                result = {
                    "request_id": response.request_id,
                    "ok": response.ok,
                    "metadata": dict(response.metadata),
                    "error_code": response.error_code,
                }
            except Exception as exc:
                result = {
                    "request_id": "",
                    "ok": False,
                    "metadata": {},
                    "error_code": type(exc).__name__,
                    "error_detail": str(exc)[:500],
                }
            try:
                connection.send_bytes(json.dumps(
                    result, separators=(",", ":"), allow_nan=False
                ).encode("utf-8"))
            except (BrokenPipeError, EOFError, OSError):
                # One cancelled/timed-out client must not terminate the
                # persistent model worker or poison every later request.
                pass
            finally:
                connection.close()
    finally:
        listener.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--endpoint", default="@scnav_qwen3vl")
    parser.add_argument(
        "--quantization",
        choices=("int8", "int4", "bf16"),
        default="int8",
    )
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    args = parser.parse_args()
    try:
        serve(args.checkpoint, args.endpoint, args.quantization, args.max_pixels)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
