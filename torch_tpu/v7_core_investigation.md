## Problem Statement

 In v7 TPU, there are 2 cores per TPU chip. for example for a `2x2x1` toplogy in v7, there are 4 tpus, 8 tpu cores. For torch_tpu, 1 TPU really mean 1 core. So I made some changes in the last commit to make the resource count of TPU in ray count 2 cores in v7 as 2 TPUs. When I ran @torch_tpu/single_tpu_train_example.py with 4 tasks, 1 TPU per task it succeeded. But when I changed it to 4 tasks * 2 TPU per task, or 8 tasks * 1 TPU per task, both times I got the error: "pjrt_state.cc:247] PjrtBackend::GetClient failed to initialize: FAILED_PRECONDITION: GetPjrtClient failed: TPU initialization failed: No TPU devices found after filtering [repeated 3x across cluster]"

## Investigation

Core Problem: Hardware Locking vs. Logical Topology
  The fundamental constraint on TPU v7 is that each physical chip (2 cores) is represented by a single `/dev/vfio/X` device file which `libtpu` locks exclusively.

  ---

  Case 1: Distributed Training (Coordinated Mesh)
  When running with `Ray Train` (coordinated workers):
   - All 8 workers can run concurrently on a 4-chip VM.
   - workers use `libtpu` in 4D Mesh mode.
   - Ray is now configured to bypass its standard 3D chip-level visibility bounds for v7 distributed tasks, allowing the backend library (`torch_tpu`) to manage the 4D mesh configuration autonomously.
   - This works perfectly because workers coordinate access to the shared physical chips.

  ---

  Case 2: Independent Tasks (Standalone Mode)
  When running independent Ray tasks (e.g., `@ray.remote`):
   - Only **4 tasks** can run concurrently on a 4-chip VM.
   - Each task MUST be assigned at least **2 TPU resources** (1 physical chip).
   - If two independent processes try to open the same physical chip, the second one fails with `Device or resource busy`.

  How it was fixed for Independent Tasks:
   - Ray now uses **3D Standalone Mode** for non-distributed tasks on v7.
   - It maps core indices back to physical chip indices for `TPU_VISIBLE_CHIPS`.
   - It ensures each task has a unique `TPU_NAME` (to avoid SHM conflicts) and a unique `TPU_PROCESS_PORT`.
   - It unsets 4D mesh variables to ensure `libtpu` remains in standalone mode.

  Summary of Best Practices for v7
  - **Distributed**: request `resources={"TPU": 1}` per worker and use `Ray Train`. All 8 cores will be utilized.
  - **Independent**: request `resources={"TPU": 2}` per task. 4 tasks will run, each utilizing 1 core (Core 0 of their assigned chip).
