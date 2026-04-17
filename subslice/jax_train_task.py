# subslice/jax_train_task.py
import os
import sys
import subprocess
import filelock
import ray
import time
from ray.util.tpu import SlicePlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

@ray.remote
class IPRecorder:
    def __init__(self):
        self.ips = {}
    def set_ip(self, rank, ip):
        self.ips[rank] = ip
    def get_ips(self):
        return self.ips

@ray.remote(num_cpus=0, resources={"TPU": 4})
def train_task(world_rank, world_size):
    import os
    import sys
    import subprocess
    import filelock
    import ray
    import time
    from ray._private.services import get_node_ip_address
    
    # Get pod IP using socket trick to avoid binding issues with hostIP
    # In a typical K8s setup without host networking, the Pod has its own separate IP. 
    # If JAX attempts to bind to the host IP from within a container that does not own that IP, 
    # the socket bind call will fail or hang indefinitely.
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        my_ip = s.getsockname()[0]
    except Exception:
        my_ip = os.environ.get("NODE_IP")
        if not my_ip:
            from ray._private.services import get_node_ip_address
            my_ip = get_node_ip_address()
    finally:
        s.close()

    print(f"Rank {world_rank} IP: {my_ip}")
        
    recorder = ray.get_actor("ip_recorder")
    ray.get(recorder.set_ip.remote(world_rank, my_ip))
    
    # Wait for all workers to register their IPs
    while True:
        ips = ray.get(recorder.get_ips.remote())
        if len(ips) == world_size:
            break
        print(f"Rank {world_rank} waiting for {world_size} IPs, got {len(ips)}")
        time.sleep(0.5)
        
    coordinator_ip = ips[0]
    worker_hostnames = ",".join([ips[i] for i in range(world_size)])
    
    # --- TPU SUBSLICE WORKAROUND ---
    # 1. Mask Pod environment variables that cause JAX to hang or fail when running on a subslice.
    pod_vars = ["TPU_WORKER_HOSTNAMES", "TPU_TOPOLOGY", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS", "TPU_WORKER_ID"]
    for var in pod_vars:
        if var in os.environ:
            del os.environ[var]
            
    # 2. Re-set environment for a 2-host 8-TPU setup (each host has 4 chips).
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1" # 4 chips in 2x2 arrangement per host
    os.environ["TPU_HOST_BOUNDS"] = "1,2,1"          # 1x2 hosts
    os.environ["TPU_WORKER_ID"] = str(world_rank)
    os.environ["TPU_WORKER_HOSTNAMES"] = worker_hostnames
    os.environ["TPU_TOPOLOGY"] = "2x4"               # 8 chips total
    
    # 3. Ensure TPU platform is preferred.
    os.environ["JAX_PLATFORMS"] = "tpu"
    # -------------------------------

    import jax
    import jax.numpy as jnp

    print(f"Rank {world_rank} initializing JAX distributed..., coordinator: {coordinator_ip}, world_size: {world_size}, worker_hostnames: {worker_hostnames}")
    
    # Manually initialize JAX distributed.
    jax.distributed.initialize(
        coordinator_address=f"{coordinator_ip}:8888",
        num_processes=world_size,
        process_id=world_rank,
    )
    
    print(f"JAX version: {jax.__version__}")
    print(f"JAX process index: {jax.process_index()}")
    print(f"JAX process count: {jax.process_count()}")
    print(f"JAX local devices: {jax.local_devices()}")
    
    key = jax.random.PRNGKey(jax.process_index())
    X = jax.random.normal(key, (100, 1))
    noise = jax.random.normal(key, (100, 1)) * 0.1
    y = 2 * X + 1 + noise

    def linear_model(params, x):
        return x @ params['w'] + params['b']

    def loss_fn(params, x, y):
        preds = linear_model(params, x)
        return jnp.mean((preds - y) ** 2)

    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        # Simple SGD update instead of optax
        learning_rate = 0.01
        params = {k: params[k] - learning_rate * grads[k] for k in params}
        return params, loss

    # Initialize parameters.
    key, w_key, b_key = jax.random.split(key, 3)
    params = {'w': jax.random.normal(w_key, (1, 1)), 'b': jax.random.normal(b_key, (1,))}

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        params, loss = train_step(params, X, y)
        time.sleep(1)
        if jax.process_index() == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
            
    return float(loss)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    
    # Create actor to share IPs
    recorder = IPRecorder.options(name="ip_recorder").remote()
    
    world_size = 2
    
    # Use SlicePlacementGroup to reserve resources for the workers.
    # Assuming a 4x4 physical slice as per cluster spec.
    print("Reserving TPU slice...")
    slice_handle = SlicePlacementGroup(topology="4x4", accelerator_version="v6e", resources_per_bundle={"TPU": 4})
    slice_pg = slice_handle.placement_group
    
    print("Waiting for placement group to be ready...")
    ray.get(slice_pg.ready(), timeout=60)
    
    print(f"Launching {world_size} training tasks on the placement group...")
    futures = [
        train_task.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=slice_pg,
            )
        ).remote(i, world_size) 
        for i in range(world_size)
    ]
    
    results = ray.get(futures)
    print(f"Training finished. Final losses: {results}")
    
    # Cleanup placement group
    slice_handle.shutdown()
