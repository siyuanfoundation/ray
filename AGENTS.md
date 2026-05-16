# Test Ray Core Changes on TPU
This fork is for adding a new torch_tpu backend (souce code in `../tmp/torch_tpu` directory) for PyTorch in Ray core, to enable Jax-free TPU training & inference like PyTorch does on GPU and CPU.

## Testing Instructions
The following commands should be run from the root of the current ray repo.

- use `./torch_tpu/build_and_push.sh` to build and push the ray image after local changes. This could take 20-40 minutes.
- use `kubectl apply -f torch_tpu/ray-tpu-single-host-v7.yaml` to deploy a new Ray cluster, with the new ray image.
- use `kubectl delete -f torch_tpu/ray-tpu-single-host-v7.yaml` to delete a Ray cluster before deploying a new one, to avoid resource conflicts.
- use `kubectl port-forward service/ray-tpu-singlehost-cluster-head-svc 8265:8265` to access the Ray cluster locally.
- submit the test script to the GKE Ray cluster with command like `ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu -- python dist_train_example_v7.py`.


## Workflow

1. Make changes in your local repo, and use `./torch_tpu/build_and_push.sh` to build and push the new image.
2. Delete the old cluster with `kubectl delete -f torch_tpu/ray-tpu-single-host-v7.yaml`.
3. Create a new Ray cluster with `kubectl apply -f torch_tpu/ray-tpu-single-host-v7.yaml`.
4. Port forward the Ray dashboard service to local: `kubectl port-forward service/ray-tpu-singlehost-cluster-head-svc 8265:8265`.
5. Submit the test script to the GKE Ray cluster with command like `ray job submit --address="http://127.0.0.1:8265" --working-dir ./torch_tpu -- python dist_train_example_v7.py`.
6. If the job fails, debug the issue, go back to step 1 to make changes, and repeat the process.
