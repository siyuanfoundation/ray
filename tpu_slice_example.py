import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.tpu import slice_placement_group

# Reserve two v6e TPU slices, each with a 4x4 topology (16 chips each).
# This topology typically has 4 VM workers, each with 4 chips.
slice_handle = slice_placement_group(
    topology="4x4", accelerator_version="v6e", num_slices=1
)
slice_pg = slice_handle.placement_group

print("Waiting for placement group to be ready...")
ray.get(slice_pg.ready(), timeout=600)  # Increased timeout for potential scaling
print("Placement group ready.")


@ray.remote(num_cpus=0, resources={"TPU": 4})
def spmd_task(world_size, rank):
    pod_name = ray.util.tpu.get_current_pod_name()
    chips_on_node = ray.util.tpu.get_num_tpu_chips_on_node()
    print(
        f"Worker Rank {rank}/{world_size}: Running on slice '{pod_name}' with {chips_on_node} chips."
    )
    return rank


# Launch one task per VM in the reserved slices. The num_workers field describes the total
# number of VMs across all slices in the SlicePlacementGroup.
tasks = [
    spmd_task.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=slice_pg,
        )
    ).remote(world_size=slice_handle.num_workers, rank=i)
    for i in range(slice_handle.num_workers)
]

results = ray.get(tasks)
print(f"Task results: {results}")
