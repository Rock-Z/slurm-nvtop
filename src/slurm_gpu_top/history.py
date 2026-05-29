from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Optional, Tuple

from .models import ClusterSnapshot, GPUDevice


GpuKey = Tuple[str, str]


@dataclass
class UtilizationHistory:
    maxlen: int = 120
    all_gpu: Deque[Optional[int]] = field(default_factory=deque)
    by_gpu: Dict[GpuKey, Deque[Optional[int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.all_gpu = deque(self.all_gpu, maxlen=self.maxlen)
        self.by_gpu = {key: deque(values, maxlen=self.maxlen) for key, values in self.by_gpu.items()}

    def record(self, snapshot: ClusterSnapshot) -> None:
        current_keys = set()
        values = []
        for gpu in _iter_gpus(snapshot):
            key = gpu_key(gpu)
            current_keys.add(key)
            value = gpu.gpu_util_percent
            self.by_gpu.setdefault(key, deque(maxlen=self.maxlen)).append(value)
            if value is not None:
                values.append(value)

        self.all_gpu.append(_average_percent(values))

        for key in list(self.by_gpu):
            if key not in current_keys:
                del self.by_gpu[key]

    def gpu_history(self, gpu: GPUDevice) -> Tuple[Optional[int], ...]:
        return tuple(self.by_gpu.get(gpu_key(gpu), ()))

    def all_history(self) -> Tuple[Optional[int], ...]:
        return tuple(self.all_gpu)


def gpu_key(gpu: GPUDevice) -> GpuKey:
    return (gpu.node, gpu.uuid or str(gpu.index))


def _iter_gpus(snapshot: ClusterSnapshot) -> Iterable[GPUDevice]:
    for node in snapshot.nodes:
        yield from node.gpus


def _average_percent(values: Iterable[int]) -> Optional[int]:
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values))
