import numpy as np
import pytest

from ray.data.util.jax_util import get_data_parallelism


def test_get_data_parallelism():
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    # Case 1: sharding is None (defaults to data parallel across all devices)
    assert get_data_parallelism(None) == jax.device_count()

    # Case 2: Not a NamedSharding
    assert get_data_parallelism("not a sharding") == 1

    # Setup Mesh
    # Use as many devices as available
    devices = jax.devices()
    num_devices = len(devices)

    # Simple 1D mesh
    mesh_1d = Mesh(np.array(devices), ("x",))

    # Case: 1D sharding along batch axis
    sharding_1d = NamedSharding(mesh_1d, P("x"))
    assert get_data_parallelism(sharding_1d) == num_devices

    # Case: 1D sharding along non-batch axis (replicated batch)
    sharding_replicated = NamedSharding(mesh_1d, P(None, "x"))
    assert get_data_parallelism(sharding_replicated) == 1

    # Setup 2D Mesh if we have at least 2 devices
    if num_devices >= 2:
        # Ensure we have an even number for a 2xN mesh if possible, or just 1xN
        d1 = 2 if num_devices % 2 == 0 else 1
        d2 = num_devices // d1
        mesh_2d = Mesh(np.array(devices).reshape((d1, d2)), ("dp", "tp"))

        # Case: 2D sharding, batch sharded along "dp"
        sharding_2d = NamedSharding(mesh_2d, P("dp", "tp"))
        assert get_data_parallelism(sharding_2d) == d1

        # Case: 2D sharding, batch sharded along BOTH axes (e.g. FSDP + TP style)
        sharding_multi = NamedSharding(mesh_2d, P(("dp", "tp"), None))
        assert get_data_parallelism(sharding_multi) == d1 * d2

        # Case: 2D sharding, batch sharded along TP axis, TP axis also shards columns
        sharding_tp_only = NamedSharding(mesh_2d, P("tp", "tp"))
        assert get_data_parallelism(sharding_tp_only) == d2

    # Case: Empty spec
    sharding_empty = NamedSharding(mesh_1d, P())
    assert get_data_parallelism(sharding_empty) == 1


def test_divisor_logic():
    # Simulate divisor calculation in jax_sync_generator
    def get_divisor(dp, hosts):
        return max(1, dp // hosts)

    # Case: Pure DP (16 total devices, 2 hosts, 8 devices/host)
    # data_parallelism = 16. divisor = 16 // 2 = 8.
    assert get_divisor(16, 2) == 8

    # Case: Replicated (16 total devices, 2 hosts, 8 devices/host)
    # data_parallelism = 1. divisor = 1.
    assert get_divisor(1, 2) == 1

    # Case: Column Sharding (DP=1)
    assert get_divisor(1, 4) == 1

    # Case: Multi-host Replication (dp=2, hosts=4)
    # Several hosts share a shard. divisor = 1.
    assert get_divisor(2, 4) == 1


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
