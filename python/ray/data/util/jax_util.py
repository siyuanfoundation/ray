from typing import Any, Dict, Iterable, Iterator, Optional, Union

import numpy as np


def get_data_parallelism(
    sharding: Optional["jax.sharding.NamedSharding"],  # noqa: F821
) -> int:
    """Returns the data parallelism degree from the named sharding.

    This is defined as the number of shards along the first dimension (batch dimension).
    """
    import jax
    from jax.sharding import NamedSharding

    if sharding is None:
        return jax.device_count()

    if not isinstance(sharding, NamedSharding):
        return 1

    mesh = sharding.mesh
    spec = sharding.spec
    if not spec:
        return 1

    batch_spec = spec[0]
    if batch_spec is None:
        return 1

    if isinstance(batch_spec, str):
        return mesh.shape[batch_spec]

    if isinstance(batch_spec, (list, tuple)):
        dp = 1
        for axis in batch_spec:
            if axis is not None:
                dp *= mesh.shape[axis]
        return dp

    return 1


def _convert_ndarray_to_jax_tensor(
    ndarray: np.ndarray,
    named_sharding: "jax.sharding.NamedSharding" = None,  # noqa: F821
) -> "jax.Array":  # noqa: F821

    local_batch_size = ndarray.shape[0]

    # Validate rank and handle partial sharding for columns with fewer dimensions.
    if named_sharding:
        from jax.sharding import NamedSharding, PartitionSpec

        partition_spec = named_sharding.spec
        if len(partition_spec) > len(ndarray.shape):
            # If the PartitionSpec has more dimensions than the array,
            # we truncate it to match the array's rank.
            # This allows 1D columns (like 'id') to still be sharded or replicated
            # even when a 2D or 3D sharding is provided for the batch.
            new_spec = partition_spec[: len(ndarray.shape)]
            named_sharding = NamedSharding(
                named_sharding.mesh, PartitionSpec(*new_spec)
            )
            partition_spec = named_sharding.spec

    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    num_hosts = jax.process_count()
    data_parallelism = get_data_parallelism(named_sharding)

    if named_sharding is None:
        # Default sharding: 1D across all devices
        global_devices = jax.devices()
        mesh = Mesh(np.array(global_devices), ("batch",))
        named_sharding = NamedSharding(mesh, PartitionSpec("batch"))
        data_parallelism = len(global_devices)

    # Calculate global batch size.
    if data_parallelism % num_hosts == 0:
        global_batch_size = local_batch_size * num_hosts
    elif num_hosts % data_parallelism == 0:
        global_batch_size = local_batch_size * data_parallelism
    else:
        # Fallback to assuming purely data parallel across all hosts
        global_batch_size = local_batch_size * num_hosts

    global_shape = (global_batch_size,) + ndarray.shape[1:]

    # Ray Data provides the full set of features (columns) on each host.
    # However, jax.make_array_from_process_local_data requires the input array
    # to match the shape of the global shard addressable by the current process.
    # We slice the ndarray to the process-addressable region.
    device_indices_map = named_sharding.addressable_devices_indices_map(global_shape)

    # Find the bounding box of indices for all local devices.
    def get_slice_start(sl: slice) -> int:
        return 0 if sl.start is None else sl.start

    def get_slice_stop(sl: slice, length: int) -> int:
        return length if sl.stop is None else sl.stop

    process_slices = []
    for d in range(len(global_shape)):
        start = min(get_slice_start(idx[d]) for idx in device_indices_map.values())
        stop = max(
            get_slice_stop(idx[d], global_shape[d])
            for idx in device_indices_map.values()
        )
        process_slices.append(slice(start, stop))

    # For the batch dimension (dim 0), the process-addressable slice is relative
    # to the global batch size. However, the input `ndarray` already contains
    # exactly the local rows for this process.
    # Thus, we translate the global batch slice to a local 0-indexed slice.
    # For all other dimensions (features), we slice the ndarray as-is.
    local_process_slices = [slice(None)] + process_slices[1:]
    ndarray = ndarray[tuple(local_process_slices)]

    return jax.make_array_from_process_local_data(named_sharding, ndarray, global_shape)


def _convert_ndarray_batch_to_jax_tensor_batch(
    ndarrays: Union[np.ndarray, Dict[str, np.ndarray]],
    named_sharding: "jax.sharding.NamedSharding" = None,  # noqa: F821
) -> Union["jax.Array", Dict[str, "jax.Array"]]:  # noqa: F821
    """Convert a NumPy ndarray batch to a globally sharded JAX Array batch.

    Args:
        ndarrays: A single NumPy ndarray or dictionary of NumPy ndarrays.
        named_sharding: The JAX NamedSharding specification defining the
            global mesh and partition layout. Default is ``None``, in which case
            the array will be sharded along the batch dimension across all devices.

    Returns:
         A globally sharded JAX Array (or dictionary of arrays) residing
         in TPU/GPU memory.
    """
    if isinstance(ndarrays, np.ndarray):
        return _convert_ndarray_to_jax_tensor(ndarrays, named_sharding)

    jax_batch = {}
    for col_name, col_ndarray in ndarrays.items():
        try:
            jax_batch[col_name] = _convert_ndarray_to_jax_tensor(
                col_ndarray, named_sharding
            )
        except ValueError as e:
            raise ValueError(f"JAX Sharding Error for column '{col_name}': \n{e}")

    return jax_batch


def jax_sync_generator(
    batch_iterable: Iterable[Any],
    drop_last: bool,
    named_sharding: "jax.sharding.NamedSharding" = None,  # noqa: F821
) -> Iterator[Union["jax.Array", Dict[str, "jax.Array"]]]:  # noqa: F821
    """A generator that synchronizes and shards batches across JAX workers.

    This generator wraps a locally yielded batch iterable and ensures that all JAX
    workers within a multi-host training setup receive the exact same number of batches
    and identical batch shapes, which is required for JAX's SPMD execution.

    It performs the following synchronizations:
    1. Checks if all workers have a batch available. If only some workers are exhausted,
       it either drops the remaining batches (`drop_last=True`) or raises an error.
    2. Finds the globally minimum local batch size across all workers.
    3. Ensures the globally minimum batch size is evenly divisible by the number of local devices.
    4. Truncates all locally yielded batches to this globally consistent minimum size.
    5. Converts the truncated local NumPy arrays into globally sharded JAX Arrays.

    Args:
        batch_iterable: An iterable yielding local data batches (either a NumPy ndarray
            or a dictionary of NumPy ndarrays).
        drop_last: If True, drops mismatched or unevenly sized leftover batches. If False,
            raises a ValueError when uneven batches or uneven batch sizes are detected.
        named_sharding: An optional JAX NamedSharding specification defining the mesh
            and partition layout. If None, the array is sharded 1D along the batch dimension.

    Yields:
        Union[jax.Array, Dict[str, jax.Array]]: A globally sharded JAX Array or a
            dictionary of JAX Arrays natively placed on devices.
    """
    import jax

    num_hosts = jax.process_count()
    data_parallelism = get_data_parallelism(named_sharding)

    # Divisor for the local batch size.
    # The local batch size must be divisible by the number of unique shards
    # handled by the current host.
    divisor = max(1, data_parallelism // num_hosts)

    iterator = iter(batch_iterable)
    while True:
        has_batch = True
        try:
            batch = next(iterator)
            if isinstance(batch, dict):
                # Use the first column to determine the batch size
                local_batch_size = len(next(iter(batch.values())))
            else:
                local_batch_size = len(batch)
        except StopIteration:
            has_batch = False
            local_batch_size = 0
            batch = None

        if num_hosts > 1:
            import jax.numpy as jnp
            from jax.experimental.multihost_utils import process_allgather

            # Synchronize batch availability and size across all hosts.
            stack = jnp.array([int(has_batch), local_batch_size], dtype=jnp.int32)
            gathered = process_allgather(stack)

            all_have_batch = bool(gathered[:, 0].all())
            any_have_batch = bool(gathered[:, 0].any())

            if not any_have_batch:
                # All workers have exhausted their data.
                break

            if not all_have_batch:
                # Some workers have exhausted their data while others have more.
                if drop_last:
                    # Drop the remaining batches from the workers that still have data.
                    break
                else:
                    # Raise an error because the remaining batches will be unevenly distributed.
                    raise ValueError(
                        "Uneven number of batches detected across JAX workers. "
                        "Some workers have exhausted their data while others have more. "
                        "To safely drop orphaned batches without hanging, "
                        "set `drop_last=True` in `iter_jax_batches()`."
                    )

            # In Case 3 (hosts % dp == 0), several hosts provide the same data.
            # They MUST provide the SAME batch size.
            if num_hosts % data_parallelism == 0 and num_hosts > data_parallelism:
                num_hosts_per_shard = num_hosts // data_parallelism
                for i in range(data_parallelism):
                    shard_batch_sizes = gathered[
                        i * num_hosts_per_shard : (i + 1) * num_hosts_per_shard, 1
                    ]
                    if not (shard_batch_sizes == shard_batch_sizes[0]).all():
                        raise ValueError(
                            f"Hosts responsible for shard {i} produced different batch sizes: "
                            f"{shard_batch_sizes}. They must provide identical data."
                        )

            min_batch_size = int(gathered[:, 1].min())
            max_batch_size = int(gathered[:, 1].max())
            # Fail all workers if any worker has a different batch size
            if max_batch_size > min_batch_size and not drop_last:
                raise ValueError(
                    "Uneven batch sizes detected across JAX workers. "
                    f"This host produced a batch of size {local_batch_size}, "
                    f"but the globally minimum batch size is {min_batch_size}. "
                    "To safely truncate the batch to the minimum size, "
                    "set `drop_last=True` in `iter_jax_batches()`."
                )
        else:
            min_batch_size = local_batch_size

        if min_batch_size % divisor != 0:
            if drop_last:
                # Align the minimum batch size to be divisible by the calculated divisor
                min_batch_size = min_batch_size - (min_batch_size % divisor)
            else:
                raise ValueError(
                    f"The globally minimum batch size ({min_batch_size}) must be evenly "
                    f"divisible by the required divisor ({divisor}) for the given sharding. "
                    f"To safely truncate the batch to a divisible size, "
                    f"set `drop_last=True` in `iter_jax_batches()`."
                )

        if min_batch_size == 0:
            # Data insufficient for even a single row across devices, skip and drop
            break

        # At this point, if local_batch_size > min_batch_size, drop_last must be True
        if local_batch_size > min_batch_size:
            # Truncate to the minimum batch size across all hosts
            if isinstance(batch, dict):
                batch = {k: v[:min_batch_size] for k, v in batch.items()}
            else:
                batch = batch[:min_batch_size]

        yield _convert_ndarray_batch_to_jax_tensor_batch(
            batch, named_sharding=named_sharding
        )
