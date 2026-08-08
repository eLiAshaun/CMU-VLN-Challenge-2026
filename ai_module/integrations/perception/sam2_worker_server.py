#!/usr/bin/env python3
"""Persistent abstract-socket SAM2 box-to-mask service."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener

from .sam2_box_worker import SAM2BoxService


def serve(
    *,
    checkpoint: str,
    config: str,
    endpoint: str,
    device: str,
) -> None:
    if not endpoint.startswith("@") or len(endpoint) < 2:
        raise ValueError("endpoint_must_be_abstract_unix_socket")
    service = SAM2BoxService(
        sam2_config=config,
        sam2_checkpoint=checkpoint,
        device=device,
    )
    listener = Listener("\0" + endpoint[1:], family="AF_UNIX")
    try:
        while True:
            connection = listener.accept()
            request_id = ""
            try:
                payload = json.loads(connection.recv_bytes().decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("request_must_be_object")
                request_id = str(payload["request_id"])
                operation = str(payload["operation"])
                if operation == "healthcheck":
                    metadata = service.healthcheck()
                elif operation == "segment_boxes":
                    request = payload.get("parameters")
                    if not isinstance(request, dict):
                        raise ValueError("sam2_parameters_must_be_object")
                    metadata = service.segment(request)
                else:
                    raise ValueError("sam2_operation_invalid")
                response = {"request_id": request_id, "ok": True, "metadata": metadata}
            except Exception as exc:
                response = {
                    "request_id": request_id,
                    "ok": False,
                    "metadata": {"error_detail": str(exc)[:500]},
                    "error_code": type(exc).__name__,
                }
            try:
                connection.send_bytes(json.dumps(
                    response,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8"))
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                connection.close()
    finally:
        listener.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--endpoint", default="@mast3r_sam2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        serve(
            checkpoint=args.checkpoint,
            config=args.config,
            endpoint=args.endpoint,
            device=args.device,
        )
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
