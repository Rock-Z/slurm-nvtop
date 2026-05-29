from __future__ import annotations

from slurm_gpu_top.history import UtilizationHistory
from slurm_gpu_top.models import ClusterSnapshot, GPUDevice, NodeSnapshot


def test_history_tracks_all_gpu_average_and_per_gpu_samples():
    history = UtilizationHistory(maxlen=4)
    first = ClusterSnapshot(
        nodes=(
            NodeSnapshot(
                node="n1",
                gpus=(
                    _gpu("n1", 0, "GPU-a", 10),
                    _gpu("n1", 1, "GPU-b", 90),
                ),
            ),
        ),
        generated_at=1,
    )
    second = ClusterSnapshot(
        nodes=(
            NodeSnapshot(
                node="n1",
                gpus=(
                    _gpu("n1", 0, "GPU-a", 20),
                    _gpu("n1", 1, "GPU-b", None),
                ),
            ),
        ),
        generated_at=2,
    )

    history.record(first)
    history.record(second)

    assert history.all_history() == (50, 20)
    assert history.gpu_history(_gpu("n1", 0, "GPU-a", 0)) == (10, 20)
    assert history.gpu_history(_gpu("n1", 1, "GPU-b", 0)) == (90, None)
    assert history.node_util_histories() == {"n1": (50, 20)}
    assert history.node_mem_histories() == {"n1": (0, 0)}


def test_history_prunes_ended_gpus_and_respects_maxlen():
    history = UtilizationHistory(maxlen=2)
    history.record(ClusterSnapshot(nodes=(NodeSnapshot(node="n1", gpus=(_gpu("n1", 0, "GPU-a", 10),)),)))
    history.record(ClusterSnapshot(nodes=(NodeSnapshot(node="n1", gpus=(_gpu("n1", 0, "GPU-a", 20),)),)))
    history.record(ClusterSnapshot(nodes=(NodeSnapshot(node="n2", gpus=(_gpu("n2", 0, "GPU-c", 30),)),)))

    assert history.all_history() == (20, 30)
    assert ("n1", "GPU-a") not in history.by_gpu
    assert "n1" not in history.by_node_util
    assert "n1" not in history.by_node_mem
    assert history.gpu_history(_gpu("n2", 0, "GPU-c", 0)) == (30,)
    assert history.node_util_histories() == {"n2": (30,)}


def test_history_tracks_node_memory_averages():
    history = UtilizationHistory(maxlen=4)
    history.record(
        ClusterSnapshot(
            nodes=(
                NodeSnapshot(
                    node="n1",
                    gpus=(
                        _gpu("n1", 0, "GPU-a", 10, mem_used=25, mem_total=100),
                        _gpu("n1", 1, "GPU-b", 90, mem_used=75, mem_total=100),
                    ),
                ),
            ),
        ),
    )

    assert history.node_mem_histories() == {"n1": (50,)}


def _gpu(
    node: str,
    index: int,
    uuid: str,
    util: int | None,
    *,
    mem_used: int = 0,
    mem_total: int = 100,
) -> GPUDevice:
    return GPUDevice(
        node=node,
        index=index,
        uuid=uuid,
        name="NVIDIA A100",
        gpu_util_percent=util,
        mem_util_percent=0,
        mem_used_mib=mem_used,
        mem_total_mib=mem_total,
        temperature_c=30,
        power_draw_w=50.0,
        power_limit_w=400.0,
    )
