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
    by_node_util: Dict[str, Deque[Optional[int]]] = field(default_factory=dict)
    by_node_mem: Dict[str, Deque[Optional[int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.all_gpu = deque(self.all_gpu, maxlen=self.maxlen)
        self.by_gpu = {key: deque(values, maxlen=self.maxlen) for key, values in self.by_gpu.items()}
        self.by_node_util = {key: deque(values, maxlen=self.maxlen) for key, values in self.by_node_util.items()}
        self.by_node_mem = {key: deque(values, maxlen=self.maxlen) for key, values in self.by_node_mem.items()}

    def record(self, snapshot: ClusterSnapshot) -> None:
        current_keys = set()
        current_nodes = set()
        values = []
        node_util_values: Dict[str, list[int]] = {}
        node_mem_values: Dict[str, list[int]] = {}
        for gpu in _iter_gpus(snapshot):
            key = gpu_key(gpu)
            current_keys.add(key)
            current_nodes.add(gpu.node)
            value = gpu.gpu_util_percent
            self.by_gpu.setdefault(key, deque(maxlen=self.maxlen)).append(value)
            if value is not None:
                values.append(value)
                node_util_values.setdefault(gpu.node, []).append(value)
            mem_value = _memory_percent(gpu)
            if mem_value is not None:
                node_mem_values.setdefault(gpu.node, []).append(mem_value)

        self.all_gpu.append(_average_percent(values))
        for node in current_nodes:
            self.by_node_util.setdefault(node, deque(maxlen=self.maxlen)).append(
                _average_percent(node_util_values.get(node, ())),
            )
            self.by_node_mem.setdefault(node, deque(maxlen=self.maxlen)).append(
                _average_percent(node_mem_values.get(node, ())),
            )

        for key in list(self.by_gpu):
            if key not in current_keys:
                del self.by_gpu[key]
        for node in list(self.by_node_util):
            if node not in current_nodes:
                del self.by_node_util[node]
        for node in list(self.by_node_mem):
            if node not in current_nodes:
                del self.by_node_mem[node]

    def gpu_history(self, gpu: GPUDevice) -> Tuple[Optional[int], ...]:
        return tuple(self.by_gpu.get(gpu_key(gpu), ()))

    def all_history(self) -> Tuple[Optional[int], ...]:
        return tuple(self.all_gpu)

    def node_util_histories(self) -> Dict[str, Tuple[Optional[int], ...]]:
        return {node: tuple(values) for node, values in self.by_node_util.items()}

    def node_mem_histories(self) -> Dict[str, Tuple[Optional[int], ...]]:
        return {node: tuple(values) for node, values in self.by_node_mem.items()}


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


def _memory_percent(gpu: GPUDevice) -> Optional[int]:
    if gpu.mem_used_mib is not None and gpu.mem_total_mib:
        return round(100 * gpu.mem_used_mib / gpu.mem_total_mib)
    return gpu.mem_util_percent
