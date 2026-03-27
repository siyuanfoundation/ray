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

# 1. Reserve one v6e TPU slice (4x4, 16 chips, 4 VMs).
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
        devices = jax.devices()
        self.mesh = Mesh(devices, axis_names=("data",))

        # Initialize parameters for binary prediction (Logistic Regression)
        input_dim = 8
        output_dim = 1
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        self.param_sharding = NamedSharding(self.mesh, P())
        self.W = jax.device_put(
            jax.random.normal(k1, (input_dim, output_dim)), self.param_sharding
        )
        self.b = jax.device_put(
            jax.random.normal(k2, (output_dim,)), self.param_sharding
        )

        print(f"Worker {self.rank}: JAX initialized.")

    def run_inference(self, ds_shard):
        import jax
        import numpy as np

        # Logistic Regression Model
        @jax.jit
        def model_fn(x, W, b):
            logits = jnp.matmul(x, W) + b
            probs = jax.nn.sigmoid(logits)
            return (probs > 0.5).astype(jnp.int32)

        print(f"Worker {self.rank}: Starting inference...")

        # batch_size=4 rows per chip -> global batch size 64 (16 chips)
        batch_iterator = ds_shard.iter_jax_batches(
            batch_size=4, synchronize_batches=True, drop_last=True
        )

        data_sharding = NamedSharding(self.mesh, P("data"))

        local_predictions = []
        for i, batch in enumerate(batch_iterator):
            x = jax.device_put(batch["features"], data_sharding)
            preds = model_fn(x, self.W, self.b)

            # Each worker returns ONLY its local addressable data.
            # On a VM with 4 chips, preds.addressable_shards contains 4 shards.
            # We concatenate them to get the data corresponding to this host.
            host_shard = np.concatenate(
                [np.array(s.data) for s in preds.addressable_shards], axis=0
            )
            local_predictions.append(host_shard)

            if i % 2 == 0:
                print(f"Worker {self.rank}: Processed batch {i}...")

        return np.concatenate(local_predictions, axis=0)


# 2. Launch workers
workers = [
    JAXInferenceWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=slice_pg)
    ).remote(rank=i, world_size=num_workers)
    for i in range(num_workers)
]

# 3. Coordinate JAX setup
worker_ips = ray.get([w.get_ip.remote() for w in workers])
coordinator_address = worker_ips[0]
ray.get([w.initialize_jax.remote(coordinator_address) for w in workers])

# 4. Prepare data (8 features)
print("Preparing dataset...")


def generate_data(batch):
    return {"features": np.random.randn(len(batch["id"]), 8).astype(np.float32)}


# 64 rows total -> exactly 1 global batch of 64.
# Each of the 4 workers gets 16 rows.
ds = ray.data.range(64).map_batches(generate_data)
shards = ds.split(num_workers, equal=True)

# 5. Execute inference
print("Launching inference tasks...")
results = ray.get(
    [workers[i].run_inference.remote(shards[i]) for i in range(num_workers)]
)

# 6. Show results (Aggregated on the driver)
# results is a list of 4 numpy arrays, each of shape (16, 1)
all_preds = np.concatenate(results, axis=0)

print("\n--- Inference Results (Binary Predictions) ---")
print(f"Total predictions collected: {len(all_preds)}")
print(f"Worker result shapes: {[r.shape for r in results]}")
print("Sample predictions (first 20):")
print(all_preds[:20].flatten())

# Cleanup
slice_handle.shutdown()
