"""Fresh episode evidence and process telemetry; no source/asset hashes."""
from __future__ import annotations

import json
from pathlib import Path
import resource
import threading
import time
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class RunLog:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.started = time.monotonic()
        self._written_crops = {}
        self._crop_serial = 0

    def store_snapshot(self, store) -> dict:
        """Keep image pixels in image files, not millions of JSON integers."""
        from PIL import Image

        snapshot = store.snapshot()
        crop_directory = self.directory / 'object_crops'
        crop_directory.mkdir(exist_ok=True)
        for oid, record in store.records.items():
            metadata = snapshot['records'][oid]['representative_crops']
            for index, crop in enumerate(record.representative_crops):
                key = (oid, index)
                previous = self._written_crops.get(key)
                if previous is None or previous[0] is not crop:
                    self._crop_serial += 1
                    relative = f'object_crops/{oid}_{index:02d}_{self._crop_serial:06d}.jpg'
                    Image.fromarray(crop).save(self.directory / relative, quality=92)
                    self._written_crops[key] = (crop, relative)
                else:
                    relative = previous[1]
                metadata[index]['path'] = relative
        return snapshot

    def event(self, event: str, **fields: Any) -> None:
        record = {'event': event, 'wall_time': time.time(),
                  'elapsed_seconds': time.monotonic() - self.started,
                  'host_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  **fields}
        with self.lock, (self.directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, default=json_value, allow_nan=False) + '\n')

    def write(self, name: str, value: Any) -> None:
        with self.lock:
            path = self.directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + '\n')
            temporary.replace(path)
