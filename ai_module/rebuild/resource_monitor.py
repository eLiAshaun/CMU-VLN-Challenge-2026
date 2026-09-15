"""Sample whole-board and per-process GPU usage alongside host memory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def nvidia_query(arguments: list[str]) -> list[str]:
    result = subprocess.run(['nvidia-smi', *arguments, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=5)
    return result.stdout.strip().splitlines() if result.returncode == 0 else [result.stderr.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--duration', type=float, default=600)
    parser.add_argument('--interval', type=float, default=2)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open('a', buffering=1) as stream:
        while time.monotonic()-started < args.duration:
            memory = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                key, value = line.split(':', 1)
                if key in {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}:
                    memory[key + '_kib'] = int(value.split()[0])
            record = {
                'time': time.time(), 'elapsed_seconds': time.monotonic()-started,
                'gpu_columns': ['name', 'memory.total MiB', 'memory.used MiB', 'utilization.gpu %'],
                'gpu': nvidia_query(['--query-gpu=name,memory.total,memory.used,utilization.gpu']),
                'process_columns': ['pid', 'process_name', 'used_memory MiB'],
                'processes': nvidia_query(['--query-compute-apps=pid,process_name,used_memory']),
                'host': memory,
            }
            stream.write(json.dumps(record) + '\n')
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
