# Design Doc: Priority-Aware Worker Group Selection in Ray Autoscaler v2

## Introduction

This document proposes an enhancement to the Ray Autoscaler (v2) to support explicit priority-based selection for worker groups. The goal is to provide users with granular control over which worker groups are preferred for scale-up operations when multiple groups satisfy the same resource demands.

## Problem Statement

The Ray Autoscaler (v2) currently employs a deterministic tie-breaking mechanism: when multiple worker groups yield the same utilization score, it selects the first group encountered in the `workerGroupSpecs` manifest. While functional, this approach presents several limitations:

1.  **Lack of Explicit Priority Control**: Users cannot define a preference hierarchy for worker groups offering similar resources (e.g., prioritizing "On-Demand" over "Spot" when cost is not the primary factor, or vice-versa).
2.  **Manifest Order Dependency**: The selection logic is implicitly tied to the order of elements in the manifest. This is fragile and difficult to manage when worker groups are added or removed dynamically (e.g., via `kubectl ray` plugins).
3.  **Ineffective Fallback Recovery**: If a preferred group experiences a temporary allocation failure, it is penalized. Under the current implementation, this penalty can become effectively permanent; even after the preferred group recovers, a secondary group with a "perfect" (zero-failure) record will always be chosen over it, regardless of the user's actual preference.

## Goals

*   Introduce an explicit, user-configurable `priority` field for worker groups.
*   Ensure that higher-priority groups are preferred for scaling when utilization scores are tied.
*   Implement a standardized recovery window that allows penalized groups to return to a "ready" state after a failure.
*   Preserve failure recency as a final tie-breaker to maintain deterministic behavior among equal-priority groups.

## Proposed Design

The solution involves making `priority` a first-class attribute in the autoscaler's scheduling and scoring logic.

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
        recoverable_availabilities.get(r.node.node_type, 1.0), # 2. Recovery Status (0.0 to 1.0)
        r.node.priority,                                      # 3. Administrative Priority
        cloud_resource_availabilities.get(r.node.node_type, 1.0), # 4. Failure Recency
    ),
    reverse=True,
)
```

**Selection Hierarchy**:
1.  **Utilization**: The primary goal remains maximizing the number of scheduled requests per node.
2.  **Recovery Status**: Nodes that have transitioned out of a failure penalty window (recovered to `1.0`) are preferred. All nodes follow a uniform recovery slope.
3.  **Administrative Priority**: Among "ready" nodes, the one with the highest user-defined `priority` is selected.
4.  **Failure Recency**: If all other factors are equal, the node with the oldest failure (or no failure) is chosen. This is calculated via the existing recency-based scoring, ensuring backward compatibility.

### 3. Penalty and Recovery Mechanism

The `CloudResourceMonitor` will be enhanced to provide two distinct views of resource availability:

#### Priority-Neutral Recovery Slope
A new function, `get_recoverable_resource_availabilities`, will calculate a continuous recovery score from 0.0 to 1.0.

*   **Recovery Window**: Controlled by the `RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S` environment variable (default: 60s). This is the window during which the score linearly recovers from `0.0` to `1.0`.
*   **Safety Floor**: A hard 5-second window (or 10% of the recovery window) during which the score remains `0.0` to prevent rapid re-launching after a failure.
*   **Formula**: `score = 0.0` if `t < Safety Floor`, else `min(1.0, t / RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S)`.

#### Failure Recency Logic
The existing `get_resource_availabilities` logic (to be exposed as `cloud_resource_availabilities`) remains unchanged. It provides a relative ranking based on the global most-recent failure. This ensures that even after the recovery window has passed, a node that failed 10 minutes ago is still preferred over a node that failed 2 minutes ago, provided they have the same priority.

## Implementation Plan

1.  **Schema Update**: Add `priority` to `NodeTypeConfig` in `python/ray/autoscaler/v2/instance_manager/config.py`.
2.  **Data Propagation**: Update `SchedulingNode` in `python/ray/autoscaler/v2/scheduler.py` to include the `priority` field.
3.  **Monitor Enhancement**:
    *   Rename `get_resource_availabilities` to `get_recency_resource_availabilities`.
    *   Implement `get_recoverable_resource_availabilities` using `RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S`.
4.  **Scheduler Update**: Refactor `_sched_best_node` in `python/ray/autoscaler/v2/scheduler.py` to utilize the new 4-level sorting key.
5.  **Integration**: Update the KubeRay operator to expose the `priority` field in the `RayCluster` CRD and pass it to the autoscaler.
