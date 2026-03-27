import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tpu import slice_placement_group

# Connect to the Ray cluster.
if not ray.is_initialized():
    ray.init()

# 1. Reserve one v6e TPU slice with 4x4 topology (16 chips total).
print("Reserving TPU slice...")
slice_handle = slice_placement_group(
    topology="4x4",
    accelerator_version="v6e",
    num_slices=1,
    resources_per_bundle={"TPU": 4},
)
slice_pg = slice_handle.placement_group

print("Waiting for placement group to be ready...")
ray.get(slice_pg.ready(), timeout=600)
print("Placement group ready.")

num_workers = slice_handle.num_bundles
print(f"Total workers in slice: {num_workers}")


@ray.remote(num_cpus=1, resources={"TPU": 4})
class JAXInferenceWorker:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.ip = ray.util.get_node_ip_address()

    def get_ip(self):
        return self.ip

    def initialize_jax(self, coordinator_address):
        jax.distributed.initialize(
            coordinator_address=f"{coordinator_address}:1234",
            num_processes=self.world_size,
            process_id=self.rank,
        )
        # Create a simple 1D mesh across all 16 devices
        devices = jax.devices()
        self.mesh = Mesh(devices, axis_names=("data",))

        # Initialize model parameters: y = xW + b
        input_dim = 32
        output_dim = 8
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        # Replicate parameters across the data axis
        self.param_sharding = NamedSharding(self.mesh, P())
        self.W = jax.device_put(
            jax.random.normal(k1, (input_dim, output_dim)), self.param_sharding
        )
        self.b = jax.device_put(
            jax.random.normal(k2, (output_dim,)), self.param_sharding
        )

        print(f"Worker {self.rank}: JAX initialized. Total devices: {len(devices)}")

    def run_inference(self, ds_shard):
        import jax

        # Linear Regression Model: y = xW + b
        @jax.jit
        def model_fn(x, W, b):
            return jnp.matmul(x, W) + b

        print(f"Worker {self.rank}: Starting inference...")

        # iter_jax_batches shards data across all devices.
        # batch_size=16 rows per host -> global batch size 64.
        batch_iterator = ds_shard.iter_jax_batches(
            batch_size=16, synchronize_batches=True, drop_last=True
        )

        # Define sharding for the input data: shard across the 'data' axis
        data_sharding = NamedSharding(self.mesh, P("data"))

        processed_count = 0
        for i, batch in enumerate(batch_iterator):
            # Explicitly put x on the global mesh with the desired sharding
            x = jax.device_put(batch["features"], data_sharding)

            # Run the model
            prediction = model_fn(x, self.W, self.b)

            # Progress tracking on rank 0
            if self.rank == 0 and i % 5 == 0:
                print(
                    f"Batch {i}: Global input shape {x.shape}, Output shape {prediction.shape}"
                )
                # We can compute a diagnostic metric (must be done on all hosts or wrapped in jit)
                # But for simple logging, checking the shape is safe.

            processed_count += x.shape[0]

        return processed_count


# 2. Launch workers
workers = [
    JAXInferenceWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=slice_pg,
        )
    ).remote(rank=i, world_size=num_workers)
    for i in range(num_workers)
]

# 3. Coordinate JAX setup
worker_ips = ray.get([w.get_ip.remote() for w in workers])
coordinator_address = worker_ips[0]
ray.get([w.initialize_jax.remote(coordinator_address) for w in workers])

# 4. Prepare data with 32 features
print("Preparing synthetic dataset...")


def generate_data(batch):
    return {"features": np.random.randn(len(batch["id"]), 32).astype(np.float32)}


ds = ray.data.range(1024).map_batches(generate_data)
shards = ds.split(num_workers, equal=True)

# 5. Execute inference
print("Launching inference tasks...")
results = ray.get(
    [workers[i].run_inference.remote(shards[i]) for i in range(num_workers)]
)

# 6. Cleanup
total = sum(results)
print(f"Inference complete! Total global rows processed: {total}")
slice_handle.shutdown()
