from unittest.mock import patch

import pytest

from ray.util.tpu import get_num_ready_tpu_slices, get_tpu_worker_resources


def test_v7_2x_verification():
    # Topology 2x2x1 has 4 nominal chips.
    topology = "2x2x1"
    # v7x pod type accounts for 2 cores per chip, so 4 chips -> v7x-8.
    accelerator_type = "tpu7x-8"

    # 1. Test get_tpu_worker_resources
    # Mock ray.nodes() to report 8 TPUs per node.
    # Requesting 8 TPUs for a 4-chip topology should work for v7 (mapping chiplets to TPUs).
    mock_nodes = [
        {
            "NodeID": "node1",
            "Alive": True,
            "Resources": {"TPU": 8},
            "Labels": {
                "ray.io/tpu-pod-type": "v7x-8",
                "ray.io/tpu-slice-name": "slice1",
                "ray.io/tpu-worker-id": "0",
            },
        }
    ]
    with patch("ray.is_initialized", return_value=True):
        with patch.dict("os.environ", {"RAY_TPU_V7_RESOURCE_IS_CORES": "True"}):
            with patch("ray.nodes", return_value=mock_nodes):
                num_workers, resources = get_tpu_worker_resources(
                    topology=topology,
                    accelerator_type=accelerator_type,
                    resources_per_unit={"TPU": 8},
                )
                assert num_workers == 1
                assert resources["TPU"] == 8

    # 2. Test get_num_ready_tpu_slices
    with patch("ray.is_initialized", return_value=True):
        with patch.dict("os.environ", {"RAY_TPU_V7_RESOURCE_IS_CORES": "True"}):
            with patch("ray.nodes", return_value=mock_nodes):
                with patch(
                    "ray._private.state.available_resources_per_node",
                    return_value={"node1": {"TPU": 8}},
                ):
                    num_slices = get_num_ready_tpu_slices(topology, accelerator_type)
                    assert num_slices == 1


def test_v7_1x_verification():
    topology = "2x2x1"
    accelerator_type = "tpu7x-8"

    # 1. Test get_tpu_worker_resources
    # Mock ray.nodes() to report only 4 TPUs per node (nominal/legacy).
    # If we request 4 TPUs per unit, we should get 1 worker (no doubling).
    mock_nodes = [
        {
            "NodeID": "node1",
            "Alive": True,
            "Resources": {"TPU": 4},
            "Labels": {
                "ray.io/tpu-pod-type": "v7x-8",
                "ray.io/tpu-slice-name": "slice1",
                "ray.io/tpu-worker-id": "0",
            },
        }
    ]
    with patch("ray.is_initialized", return_value=True):
        with patch("ray.nodes", return_value=mock_nodes):
            num_workers, resources = get_tpu_worker_resources(
                topology=topology,
                accelerator_type=accelerator_type,
                resources_per_unit={"TPU": 4},
            )
            assert num_workers == 1
            assert resources["TPU"] == 4

    # 2. Test get_num_ready_tpu_slices
    # Node reports 4 TPU resources (nominal chips) but STILL has the standard v7x-8 pod type label.
    with patch("ray.is_initialized", return_value=True):
        with patch("ray.nodes", return_value=mock_nodes):
            with patch(
                "ray._private.state.available_resources_per_node",
                return_value={"node1": {"TPU": 4}},
            ):
                num_slices = get_num_ready_tpu_slices(topology, accelerator_type)
                assert num_slices == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-sv", __file__]))
