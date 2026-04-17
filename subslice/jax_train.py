# ray job submit --address="http://127.0.0.1:8265" --working-dir ./subslice -- python jax_train.py

import os
import ray.train
from ray.train.v2.api.data_parallel_trainer import DataParallelTrainer
from ray.train import ScalingConfig

def train_func():
    """This function is run on each distributed worker."""
    import os
    import sys
    import subprocess
    import filelock
    
    # --- TPU SUBSLICE WORKAROUND ---
    # 1. Mask Pod environment variables that cause JAX to hang or fail when running on a subslice.
    # These variables are often set by the TPU VM or KubeRay to describe the full 4x4 topology.
    pod_vars = ["TPU_WORKER_HOSTNAMES", "TPU_TOPOLOGY", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS", "TPU_WORKER_ID"]
    for var in pod_vars:
        if var in os.environ:
            del os.environ[var]
    
    # 2. Re-set environment for a standalone single-host 4-TPU setup (v6e).
    # This forces JAX/XLA to treat the current node as the entire cluster.
    os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "2,2,1" # 4 chips in 2x2 arrangement
    os.environ["TPU_HOST_BOUNDS"] = "1,1,1"          # 1 host
    os.environ["TPU_WORKER_ID"] = "0"                # Local worker index 0
    os.environ["TPU_WORKER_HOSTNAMES"] = "127.0.0.1" # Only local host
    os.environ["TPU_TOPOLOGY"] = "2x2"               # 4 chips total
    
    # 3. Ensure TPU platform is preferred.
    os.environ["JAX_PLATFORMS"] = "tpu"
    # -------------------------------

    # Install optax without its dependencies to preserve the base image's jax/jaxlib.
    with filelock.FileLock("/tmp/optax_install.lock"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optax", "--no-deps", "--user", "--quiet"])

    import jax
    import jax.numpy as jnp
    import optax
    
    # 4. Manually initialize JAX distributed for a single-host job.
    # This must be called before any JAX operations and bypasses multi-host coordination.
    jax.distributed.initialize(
        coordinator_address="127.0.0.1:8888",
        num_processes=1,
        process_id=0,
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
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        # time.sleep(1)
        return params, opt_state, loss

    # Initialize parameters and optimizer.
    key, w_key, b_key = jax.random.split(key, 3)
    params = {'w': jax.random.normal(w_key, (1, 1)), 'b': jax.random.normal(b_key, (1,))}
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        params, opt_state, loss = train_step(params, opt_state, X, y)
        if jax.process_index() == 0 and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
        # Report metrics back to Ray Train.
        ray.train.report({"loss": float(loss), "epoch": epoch})

# Define the TPU scaling configuration.
scaling_config = ScalingConfig(
    num_workers=1,
    use_tpu=True,
    resources_per_worker={"TPU": 4},
    accelerator_type="TPU-V6E",
)

# Define and run the Trainer.
# Using DataParallelTrainer to manually handle JAX initialization for subslice support.
trainer = DataParallelTrainer(
    train_loop_per_worker=train_func,
    scaling_config=scaling_config,
)
result = trainer.fit()
print(f"Training finished. Final metrics: {result.metrics}")
