## Problem Statement

 In v7 TPU, there are 2 cores per TPU chip. for example for a `2x2x1` toplogy in v7, there are 4 tpus, 8 tpu cores. For torch_tpu, 1 TPU really mean 1 core. So I made some changes in the last commit to make the resource count of TPU in ray count 2 cores in v7 as 2 TPUs.

 * image: `us-east5-docker.pkg.dev/tpu-vm-gke-testing/sizhang-repo/ray-torch-tpu:v7`

 - `torch_tpu/dist_train_example_v7.py` with 8 workers passes
 - `torch_tpu/single_tpu_train_example.py` with 1 1-TPU task passes
 - `torch_tpu/single_tpu_train_example.py` with 4 1-TPU tasks passes
 - `torch_tpu/single_tpu_train_example.py` with 8 1-TPU / 4 2-TPU tasks fails with
 ```
 pjrt_state.cc:247] PjrtBackend::GetClient failed to initialize: FAILED_PRECONDITION: GetPjrtClient failed: TPU initialization failed: No TPU devices found after filtering [repeated 3x across cluster]
 ```
 - `torch_tpu/single_tpu_train_example.py` with 1 or 2 2-TPU tasks passes



 ## Iteration 1

 * image: `us-east5-docker.pkg.dev/tpu-vm-gke-testing/sizhang-repo/ray-torch-tpu:v7-1`

 Setting `TPU_CHIPS_PER_HOST_BOUNDS` and `TPU_HOST_BOUNDS` from "1,1,1: to "1,1,1,1" for single chip case.

 For 2 chip case, `TPU_CHIPS_PER_HOST_BOUNDS` from "1,2,1" to "1,1,1,2", `TPU_HOST_BOUNDS` from "1,1,1" to "1,1,1,2"

 Results:

 - `torch_tpu/dist_train_example_v7.py` with 8 workers passes
 - `torch_tpu/single_tpu_train_example.py` with 1 1-TPU task passes
 - `torch_tpu/single_tpu_train_example.py` with 4 1-TPU tasks fails with
 ```
 RuntimeError: failed to enqueue execution for task_name=anonymous: Attempted to execute with 1 argument lists when local device count is 0 (total replica count: 1, partition count: 1) - TpuMemcpyDtoHDirect: DeviceBufferRef has nonzero size, but does not have a PjRtBuffer to copy from.
 ```

 - `torch_tpu/single_tpu_train_example.py` with 8 tasks fails with
 ```
 (train_linear_regression pid=8095, ip=10.56.9.10) I0000 00:00:1779049558.149474    8095 pjrt_c_api_client.cc:197] PjRtCApiClient created. [repeated 7x across cluster]
(train_linear_regression pid=8095, ip=10.56.9.10) I0000 00:00:1779049558.185243   17490 tier2_compilation_cache.cc:149] Tier-2 compilation cache is disabled for world size 1. [repeated 7x across cluster]
(train_linear_regression pid=8095, ip=10.56.9.10) I0000 00:00:1779049558.185269   17490 tier3_compilation_cache.cc:65] Backup compilation for tier-3 cache read is disabled as tier-3 cache is disabled. [repeated 7x across cluster]
(train_linear_regression pid=8092, ip=10.56.9.10) /tmp/ray/session_2026-05-17_13-23-36_590096_1/runtime_resources/working_dir_files/_ray_pkg_8c3bd11174d7c2b5/single_tpu_train_example.py:60: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at external/rules_python++pip+torch_tpu_pypi_312_torch/site-packages/torch/include/ATen/core/TensorBody.h:498.) [repeated 7x across cluster]
(train_linear_regression pid=8092, ip=10.56.9.10)   final_loss = loss.item() [repeated 3x across cluster]
(train_linear_regression pid=8092, ip=10.56.9.10)   print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}") [repeated 3x across cluster]

---------------------------------------
Job 'raysubmit_73nQ2jgcpZATWrNN' failed
---------------------------------------

Status message: Job entrypoint command failed with exit code 1, last available logs (truncated to 20,000 chars):
(pid=8095, ip=10.56.9.10) I0000 00:00:1779049554.253978    8095 device_rt.cc:88] PjRt runtime initialization deferred for tpu [repeated 7x across cluster]
(pid=8095, ip=10.56.9.10) Successfully renamed PrivateUse1 backend to 'tpu'. Device: device(type='tpu') [repeated 7x across cluster]
(pid=8095, ip=10.56.9.10) Registered Python module for 'tpu'. [repeated 7x across cluster]
(train_linear_regression pid=8095, ip=10.56.9.10) I0000 00:00:1779049554.271177    8095 pjrt_api.cc:167] The PJRT plugin has PJRT API version 0.108. The framework PJRT API version is 0.104. [repeated 7x across cluster]
(train_linear_regression pid=8095, ip=10.56.9.10) I0000 00:00:1779049558.149474    8095 pjrt_c_api_client.cc:197] PjRtCApiClient created. [repeated 7x across cluster]
(train_linear_regression pid=8095, ip=10.56.9.10) I0000 00:00:1779049558.185243   17490 tier2_compilation_cache.cc:149] Tier-2 compilation cache is disabled for world size 1. [repeated 7x across cluster]
(train_linear_regression pid=8095, ip=10.56.9.10) I0000 00:00:1779049558.185269   17490 tier3_compilation_cache.cc:65] Backup compilation for tier-3 cache read is disabled as tier-3 cache is disabled. [repeated 7x across cluster]
(train_linear_regression pid=8092, ip=10.56.9.10) /tmp/ray/session_2026-05-17_13-23-36_590096_1/runtime_resources/working_dir_files/_ray_pkg_8c3bd11174d7c2b5/single_tpu_train_example.py:60: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more information. (Triggered internally at external/rules_python++pip+torch_tpu_pypi_312_torch/site-packages/torch/include/ATen/core/TensorBody.h:498.) [repeated 7x across cluster]
(train_linear_regression pid=8092, ip=10.56.9.10)   final_loss = loss.item() [repeated 3x across cluster]
(train_linear_regression pid=8092, ip=10.56.9.10)   print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}") [repeated 3x across cluster]
 ```

- `torch_tpu/single_tpu_train_example.py` with 1 2-TPU tasks failed with
```
RuntimeError: materialization failed with: GetPjrtClient failed: TPU initialization failed: Invalid --deepsea_slice_builder_worker_addresses specified. Expected 2 worker addresses, got 1.
```
