Core Problem: Hardware vs. Logical Device Visibility
  The fundamental issue is a conflict between how the OS/Ray sees the TPU (8 separate files/resources) and how the PJRT/libtpu layers see and filter them.

  On v7 TPUs, there are 2 cores per chip. In a 4-chip system:
   - OS Level: 8 device files exist in /dev/vfio. Ray correctly detects these as 8 resources.
   - Driver Level (libtpu): The TPU_VISIBLE_CHIPS environment variable filters at the chip level, not the core level. It only accepts indices 0, 1, 2, 3.

  ---

  Case 1: 8 Tasks × 1 TPU Core per Task
  When Ray assigns logical TPU 4 to a task, it sets TPU_VISIBLE_CHIPS="4".

  Why it fails:
   - In torch_tpu/pjrt/pjrt_init.cc: The call to InitializePjrtPlugin (line 60) triggers the underlying libtpu to look for chip index 4.
   - The Error: FAILED_PRECONDITION: GetPjrtClient failed: TPU initialization failed: No TPU devices found after filtering.
   - The Cause: Chip 4 does not exist (only 0-3). Even though logical device 4 exists in /dev/vfio, the driver-level filtering mechanism (TPU_VISIBLE_CHIPS) doesn't
     recognize it.

  If we fixed the Ray mapping (mapping 4 -> Chip 2):
   - Ray sets TPU_VISIBLE_CHIPS="2".
   - torch_tpu discovers two addressable devices (Core 0 and Core 1 of Chip 2).
   - The Hardcoding: In torch_tpu/pjrt/pjrt_init.cc:

   1   // Line 100
   2   xla::PjRtDevice* device = addressable_devices[0];
   - The Result: Both Task 4 and Task 5 (which both map to Chip 2) will pick addressable_devices[0]. They will both run on the same core, causing a resource collision
     and leaving the other core idle.

  ---

  Case 2: 4 Tasks × 2 TPU Cores per Task
  When Ray assigns logical TPUs 0, 1 to a task, it sets TPU_VISIBLE_CHIPS="0,1".

  Why it fails (or behaves incorrectly):
   - In torch_tpu/pjrt/pjrt_init.cc:
     - libtpu initializes and sees two chips (0 and 1), which contain a total of four cores.
     - addressable_devices.size() will be 4.
   - The Hardcoding:

   1   // Line 100
   2   xla::PjRtDevice* device = addressable_devices[0];
   - The Problem: The task is only assigned one xla::PjRtDevice. The torch_tpu library currently has no logic to utilize multiple local PjRtDevice objects within a
     single process. It explicitly picks the first one and ignores the rest.
   - The Result: Even though the task "owns" 2 cores, it only uses 1. Furthermore, if world_size is not set to 4 (matching the discovered device count), the
     initialization will explicitly fail at:

   1   // Line 89
   2   TT_RET_CHECK(world_size == 1 || world_size == device_count, ...)
    Since device_count will be 4 (total cores visible) but the task only intends to be 1 of 4 workers, this check might fail depending on how world_size is passed.

  ---

  Summary of torch_tpu Limitations
  The torch_tpu library currently assumes a 1-process-per-chip model where the process uses exactly one PjRtDevice. It is not designed to:
   1. Select a specific core among multiple visible ones (preventing 1-core tasks from sharing a chip).
   2. Aggregate multiple local cores into a single logical device for the user (preventing 2-core tasks from utilizing their full allocation).

  Without changing torch_tpu, the only way to support 8 cores is to ensure each task has exclusive access to a physical chip (mapping logical indices to
  TPU_VISIBLE_CHIPS and selecting the correct core index 0 or 1), which requires the library to be core-aware.
