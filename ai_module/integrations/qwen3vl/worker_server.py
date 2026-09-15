#!/usr/bin/env python3
"""Abstract-Unix-socket wrapper around the local Qwen3-VL worker."""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing.connection import Listener
from pathlib import Path

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
        deadline_monotonic=float(payload.get("deadline_monotonic", 0.0)),
    )


def serve(
    checkpoint: str,
    endpoint: str,
    quantization: str = "int8",
    max_pixels: int = 512 * 512,
    batch_max_new_tokens: int = 256,
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
                batch_max_new_tokens=batch_max_new_tokens,
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

                # Check worker status before accepting request
                if request.deadline_monotonic > 0.0 and (
                    request.deadline_monotonic <= time.monotonic()
                ):
                    result = {
                        "request_id": request.request_id,
                        "ok": False,
                        "metadata": {},
                        "error_code": "DEADLINE_EXPIRED",
                    }
                else:
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
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--endpoint", required=False)
    parser.add_argument(
        "--quantization",
        choices=("int8", "int4", "bf16"),
        required=False,
    )
    parser.add_argument("--max-pixels", type=int, required=False)
    parser.add_argument(
        "--batch-max-new-tokens", type=int, required=False
    )
    args = parser.parse_args()

    # Backward compatibility: support CLI arguments
    if args.checkpoint:
        checkpoint = args.checkpoint
        endpoint = args.endpoint or "@scnav_qwen3vl"
        quantization = args.quantization or "int8"
        max_pixels = args.max_pixels or 512 * 512
        batch_max_new_tokens = args.batch_max_new_tokens or 256
    else:
        # New path: load from config
        try:
            from config import load_config
            ai_module_root = Path(__file__).resolve().parents[2]
            config = load_config(
                asset_manifest_path=ai_module_root / "configs" / "model_assets.json"
            )
            checkpoint = str(ai_module_root / config.qwen3vl.checkpoint_path)
            endpoint = config.qwen3vl.endpoint
            quantization = config.qwen3vl.quantization
            max_pixels = config.qwen3vl.max_pixels
            batch_max_new_tokens = config.qwen3vl.batch_max_new_tokens
        except ImportError:
            raise RuntimeError(
                "No --checkpoint provided and config system not available. "
                "Use --checkpoint, --endpoint, --quantization, --max-pixels"
            )

    try:
        serve(
            checkpoint,
            endpoint,
            quantization,
            max_pixels,
            batch_max_new_tokens,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
