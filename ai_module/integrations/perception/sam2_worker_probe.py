#!/usr/bin/env python3
"""Wait until the persistent SAM2 service has loaded its checkpoint."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Client
import os
import time
import uuid


def probe(endpoint: str, timeout_s: float, worker_pid: int | None = None) -> dict:
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if worker_pid is not None:
            try:
                os.kill(worker_pid, 0)
            except OSError as exc:
                raise RuntimeError("sam2_worker_exited") from exc
        connection = None
        try:
            connection = Client("\0" + endpoint[1:], family="AF_UNIX")
            request_id = uuid.uuid4().hex
            connection.send_bytes(json.dumps({
                "request_id": request_id,
                "operation": "healthcheck",
                "parameters": {},
            }, separators=(",", ":")).encode("utf-8"))
            if not connection.poll(max(0.1, deadline - time.monotonic())):
                raise TimeoutError("sam2_healthcheck_timeout")
            response = json.loads(connection.recv_bytes().decode("utf-8"))
            if response.get("request_id") != request_id or response.get("ok") is not True:
                raise RuntimeError(str(response.get("error_code") or "sam2_healthcheck_failed"))
            return dict(response["metadata"])
        except (ConnectionError, EOFError, OSError) as exc:
            last_error = exc
            time.sleep(0.2)
        finally:
            if connection is not None:
                connection.close()
    if last_error is not None:
        raise RuntimeError(f"sam2_healthcheck_unreachable:{type(last_error).__name__}") from last_error
    raise TimeoutError("sam2_healthcheck_timeout")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="@mast3r_sam2")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    print(json.dumps(probe(args.endpoint, args.timeout, args.pid), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
