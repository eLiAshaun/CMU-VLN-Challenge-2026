from __future__ import annotations

from multiprocessing import shared_memory
import numpy as np


class SharedArrayStore:
    def put(self, array: np.ndarray) -> tuple[str, tuple[int, ...], str]:
        value = np.ascontiguousarray(array)
        block = shared_memory.SharedMemory(create=True, size=value.nbytes)
        np.ndarray(value.shape, dtype=value.dtype, buffer=block.buf)[:] = value
        block.close()
        return block.name, value.shape, value.dtype.str

    def get(self, name: str, shape, dtype, *, unlink: bool = False) -> np.ndarray:
        block = shared_memory.SharedMemory(name=str(name))
        try:
            result = np.ndarray(tuple(shape), dtype=np.dtype(dtype), buffer=block.buf).copy()
        finally:
            block.close()
            if unlink:
                block.unlink()
        return result
