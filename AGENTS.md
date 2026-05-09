# Test Ray Core Changes on TPU

## Testing Instructions
The following commands should be run from the root of the current ray repo.

- use `./torch_tpu/build_and_push.sh` to build and push the ray image after local changes.
- use `kubectl apply -f torch_tpu/ray-tpu-single-host-v7.yaml` to deploy a new Ray cluster, with the new ray image.
- use `kubectl delete -f torch_tpu/ray-tpu-single-host-v7.yaml` to delete a Ray cluster before deploying a new one, to avoid resource conflicts.
- use `kubectl port-forward service/ray-tpu-singlehost-cluster-head-svc 8265:8265` to access the Ray cluster locally.
- submit the `torch_tpu/dist_train_example_v7.py` and `torch_tpu/single_tpu_train_example.py` jobs to make sure they finish successfully.
