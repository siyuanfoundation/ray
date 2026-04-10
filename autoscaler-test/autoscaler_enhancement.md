# Design Doc: Priority-Aware Worker Group Selection in Ray Autoscaler v2

## Introduction

This document proposes an enhancement to the Ray Autoscaler (v2) to support explicit priority-based selection for worker groups. The goal is to provide users with granular control over which worker groups are preferred for scale-up operations when multiple groups satisfy the same resource demands.

## Problem Statement

The Ray Autoscaler (v2) currently employs a deterministic tie-breaking mechanism: when multiple worker groups yield the same utilization score, it selects the first group encountered in the `workerGroupSpecs` manifest. While functional, this approach presents several limitations:

1.  **Lack of Explicit Priority Control**: Users cannot define an explicit preference hierarchy for worker groups offering similar resources (e.g., prioritizing "On-Demand" over "Spot" when cost is not the primary factor, or vice-versa).
2.  **Manifest Order Dependency**: The selection logic is implicitly tied to the order of elements in the manifest, and it is not officially documented. This is difficult to manage when worker groups are added or removed dynamically (e.g., via `kubectl ray` plugins). It is also fragile and can lead to unexpected behavior when the implementation details change.
3.  **Ineffective Fallback Recovery**: If a preferred group experiences a temporary allocation failure, it is penalized. Under the [current implementation](https://github.com/ray-project/ray/blob/f9ccc7a79ee4535a5575551687e193c003e6c7f9/python/ray/autoscaler/v2/instance_manager/subscribers/cloud_resource_monitor.py#L62-L73), this penalty never goes to 0, so it can become effectively permanent; even after the preferred group recovers, a secondary group with a "perfect" (zero-failure) record will always be chosen over it, regardless of the user's actual preference.

## Requirements

*   **Explicit Priority**: Introduce an optional, user-configurable `priority` field for worker groups to allow explicit preference control.
*   **Backward Compatibility**:
    *   Keep the current computation of the [utilization score](https://github.com/ray-project/ray/blob/f9ccc7a79ee4535a5575551687e193c003e6c7f9/python/ray/autoscaler/v2/scheduler.py#L476) unchanged.
    *   Preserve the default behavior when no priority is specified (i.e., priority defaults to 0).
*   **Availability Prioritized**:
    * Favor available groups over high-priority failing ones, while providing a recovery window for penalized groups.
    * When availability is equal, prefer higher priority groups.

## Proposed Design

The solution involves making `priority` a first-class attribute in the autoscaler's scheduling and scoring logic. The Deterministic Selection Hierarchy for choosing worker groups for scale-up will follow these strict, deterministic orders:
1. **Utilization**: Maximize resource utilization first.
2. **Recoverable Availability**: Prefer nodes that are currently available or have recovered from failures.
3. **Priority**: Prefer higher-priority nodes when utilization and availability are equal.
4. **Failure Recency**: Use historical failure recency as the final tie-breaker.

### 1. Configuration Schema Changes

The `priority` field will be introduced as an optional non-negative integer.

*   **RayCluster CRD**: The `workerGroupSpecs` will include an optional `priority` field (defaulting to 0).
*   **Python Autoscaler**:
    *   `NodeTypeConfig` will be updated to store the `priority` parsed from the cluster configuration.
    *   `SchedulingNode` will carry this `priority` to the scheduler.

### 2. Multi-Level Node Selection Logic

The `ResourceDemandScheduler` will be updated to use a 4-level sorting key for selecting the optimal node for a resource request. The key is evaluated in descending order of importance (higher value is better):

```python
results = sorted(
    results,
    key=lambda r: (
        r.score,                                              # 1. Resource Utilization
        recoverable_availabilities.get(r.node.node_type, 1.0), # 2. Recoverable Availability (0.0 to 1.0)
        r.node.priority,                                      # 3. Administrative Priority
        cloud_resource_availabilities.get(r.node.node_type, 1.0), # 4. Failure Recency
    ),
    reverse=True,
)
```

**Selection Hierarchy**:
1.  **Utilization**: The primary goal remains maximizing the number of scheduled requests per node.
2.  **Recoverable Availability**: Nodes that have transitioned out of a failure penalty window (recovered to `1.0`) are preferred. All nodes follow a uniform recovery slope.
3.  **Administrative Priority**: Among "ready" nodes, the one with the highest user-defined `priority` is selected.
4.  **Failure Recency**: If all other factors are equal, the node with the oldest failure (or no failure) is chosen. This is calculated via the existing recency-based scoring (`cloud_resource_availabilities`), ensuring backward compatibility.

### 3. Penalty and Recovery Mechanism

The `CloudResourceMonitor` will be enhanced to provide two distinct views of resource availability:

#### Priority-Neutral Recovery Slope
A new function, `get_recoverable_resource_availabilities`, will calculate a continuous recovery score from 0.0 to 1.0.

*   **Recovery Window**: Controlled by the `RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S` environment variable (default: 600s). This is the window during which the score linearly recovers from `0.0` to `1.0`.
*   **Safety Floor**: A hard 10-second window (or 10% of the recovery window) during which the score remains `0.0` to prevent rapid re-launching after a failure.
*   **Formula**: `score = 0.0` if `(current_time - last_unavailable_timestamp) < Safety Floor`, else `min(1.0, (current_time - last_unavailable_timestamp) / RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S)`.

#### Failure Recency Logic
The existing `get_resource_availabilities` logic (exposed as `cloud_resource_availabilities`) remains unchanged. It provides a relative ranking based on the global most-recent failure. This ensures that even after the recovery window has passed, a node that failed 10 minutes ago is still preferred over a node that failed 2 minutes ago, provided they have the same priority.

## Implementation Plan

1.  **Schema Update**: Add `priority` to `NodeTypeConfig` in `python/ray/autoscaler/v2/instance_manager/config.py`.
2.  **Data Propagation**: Update `SchedulingNode` in `python/ray/autoscaler/v2/scheduler.py` to include the `priority` field.
3.  **Monitor Enhancement**:
    *   Expose existing logic as `cloud_resource_availabilities`.
    *   Implement `get_recoverable_resource_availabilities` using `RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S`.
4.  **Scheduler Update**: Refactor `_sched_best_node` in `python/ray/autoscaler/v2/scheduler.py` to utilize the new 4-level sorting key.
5.  **Integration**: Update the KubeRay operator to expose the `priority` field in the `RayCluster` CRD and pass it to the autoscaler.

## Testing Plan

To ensure the reliability and correctness of the priority-aware selection logic, the following testing strategy will be implemented:

### 1. Unit Testing
*   **Recovery Scoring**: Test `get_recoverable_resource_availabilities` in `test_cloud_resource_monitor.py`.
    *   Verify score is `0.0` immediately after failure and remains `0.0` for the safety floor (10s).
    *   Verify score reaches `0.5` at 300s and `1.0` at 600s (with default 600s window).
*   **Sorting Logic**: Test `_sched_best_node` in `test_scheduler.py` with mock data:
    *   **Tie-breaking**: Two node types with identical resources and recovery scores. Verify the one with higher `priority` is chosen.
    *   **Fallback**: High-priority group with recovery score `< 1.0` vs. Low-priority group with recovery score `= 1.0`. Verify the low-priority group is chosen.
    *   **Recency**: Two groups with same priority and recovery score `= 1.0`. Verify the one with a higher historical availability score (older failure) is chosen.

### 2. Integration Testing
*   **Autoscaler Loop**: Use `test_resource_demand_scheduler.py` to simulate end-to-end scaling iterations with `RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S=60`.
    *   Scenario: `Group A (Pri: 10)` and `Group B (Pri: 0)`.
    *   Iteration 1: Create demand. Expect `Group A` pod.
    *   Iteration 2: Mock `Group A` allocation failure. Create demand. Expect `Group B` pod.
    *   Iteration 3: Wait 70s. Create demand. Expect `Group A` pod.

### 3. End-to-End (E2E) Testing
*   **KubeRay Experiment**: Deploy a `RayCluster` with two identical worker groups with different `priority` values.
    *   Verify that `priority` is correctly propagated to the autoscaler logs.
    *   Verify that the high-priority group is always selected first for scale-up.
    *   Simulate infrastructure constraints (e.g., using a non-existent image or invalid resource request to trigger failure) and verify the fallback and recovery behavior.
