import jax

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tpu import slice_placement_group

# Connect to the Ray cluster.
# The user mentioned http://127.0.0.1:8265 for job submission,
# so we can just use ray.init() if running on the cluster.
if not ray.is_initialized():
    ray.init()

# 1. Reserve one v6e TPU slice with 4x4 topology (16 chips, 4 hosts).
print("Reserving TPU slice...")
# We explicitly set resources_per_bundle to {"TPU": 4} because v6e 4x4 slices
# in this cluster typically have 4 VM workers with 4 chips each.
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

# Use num_bundles to ensure we launch one worker per 4-chip bundle.
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
        # All JAX hosts must call initialize for multihost SPMD.
        # Coordinator address is the IP of rank 0.
        coordinator_address_with_port = f"{coordinator_address}:1234"
        print(
            f"Worker {self.rank}: Initializing JAX distributed with coordinator {coordinator_address_with_port}"
        )
        jax.distributed.initialize(
            coordinator_address=coordinator_address_with_port,
            num_processes=self.world_size,
            process_id=self.rank,
        )
        print(
            f"Worker {self.rank}: JAX distributed initialized. Total devices: {jax.device_count()}, Local devices: {jax.local_device_count()}"
        )

    def run_inference(self, ds_shard):
        import jax

        # Simple dummy model for demo: y = x * 2
        @jax.jit
        def model_fn(x):
            return x * 2

        print(f"Worker {self.rank}: Starting inference...")

        # iter_jax_batches handles sharding across local TPU chips (4 chips per host).
        # batch_size=8 means 8 rows per host, total 32 rows across 4 hosts.
        # synchronize_batches=True ensures all hosts stay in sync.
        batch_iterator = ds_shard.iter_jax_batches(
            batch_size=8, synchronize_batches=True, drop_last=True
        )

        processed_count = 0
        for i, batch in enumerate(batch_iterator):
            # ray.data.range() produces a dict {"id": ...}
            x = batch["id"]
            prediction = model_fn(x)

            # For demo, we just print the first host's progress
            if self.rank == 0 and i % 2 == 0:
                print(f"Batch {i}: input shape {x.shape}")

            processed_count += x.shape[0]

        return processed_count


# 2. Launch workers on the TPU slice VMs.
# Each worker gets 4 TPU chips (one full VM).
workers = [
    JAXInferenceWorker.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=slice_pg,
        )
    ).remote(rank=i, world_size=num_workers)
    for i in range(num_workers)
]

# 3. Coordinate JAX distributed initialization.
# We need the IP of the first worker to act as the coordinator.
print("Gathering worker IPs...")
worker_ips = ray.get([w.get_ip.remote() for w in workers])
coordinator_address = worker_ips[0]

print(f"Coordinator address: {coordinator_address}. Initializing JAX cluster...")
ray.get([w.initialize_jax.remote(coordinator_address) for w in workers])

# 4. Prepare data and split it across workers.
# Total 128 rows, split equally into 4 shards of 32 rows each.
print("Preparing dataset...")
ds = ray.data.range(128)
shards = ds.split(num_workers, equal=True)

# 5. Execute inference on all hosts.
print("Launching inference tasks...")
inference_tasks = [
    workers[i].run_inference.remote(shards[i]) for i in range(num_workers)
]

# 6. Collect and print results.
results = ray.get(inference_tasks)
total_processed = sum(results)
print(
    f"Inference complete! Total rows processed across {num_workers} hosts: {total_processed}"
)
print(f"Per-host counts: {results}")

# 7. Cleanup
slice_handle.shutdown()
