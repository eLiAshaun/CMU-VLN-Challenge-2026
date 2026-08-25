#!/usr/bin/env python3
"""Prove that the local Qwen worker has loaded and retained its weights."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Client
import os
import time
import uuid


def probe(endpoint: str, timeout_s: float, worker_pid: int | None = None) -> dict:
    if not endpoint.startswith("@") or len(endpoint) < 2:
        raise ValueError("endpoint_must_be_abstract_unix_socket")
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if worker_pid is not None:
            try:
                os.kill(worker_pid, 0)
            except OSError as exc:
                raise RuntimeError("qwen3vl_worker_exited") from exc
        connection = None
        try:
            connection = Client("\0" + endpoint[1:], family="AF_UNIX")
            request_id = uuid.uuid4().hex
            connection.send_bytes(json.dumps({
                "request_id": request_id,
                "acquisition_id": "launch-healthcheck",
                "backend": "qwen3vl",
                "operation": "healthcheck",
                "input_handles": [],
                "parameters": {},
            }, separators=(",", ":")).encode("utf-8"))
            remaining = max(0.1, deadline - time.monotonic())
            if not connection.poll(remaining):
                raise TimeoutError("qwen3vl_healthcheck_timeout")
            response = json.loads(connection.recv_bytes().decode("utf-8"))
            if response.get("request_id") != request_id:
                raise RuntimeError("qwen3vl_healthcheck_request_mismatch")
            if not response.get("ok"):
                raise RuntimeError(
                    str(response.get("error_code") or "qwen3vl_healthcheck_failed")
                )
            metadata = dict(response.get("metadata", {}))
            if metadata.get("ready") is not True:
                raise RuntimeError("qwen3vl_healthcheck_not_ready")
            return metadata
        except (ConnectionError, EOFError, OSError) as exc:
            last_error = exc
            time.sleep(0.2)
        finally:
            if connection is not None:
                connection.close()
    if last_error is not None:
        raise RuntimeError(
            f"qwen3vl_healthcheck_unreachable:{type(last_error).__name__}"
        ) from last_error
    raise TimeoutError("qwen3vl_healthcheck_timeout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="@scnav_qwen3vl")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    print(json.dumps(
        probe(args.endpoint, args.timeout, args.pid),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
