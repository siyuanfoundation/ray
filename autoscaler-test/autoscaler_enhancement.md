# Enhance Ray Autoscaler to Set Relative Priority Order for Worker Groups

## Problem Statement

Currently, Ray Autoscaler v2 does not support setting relative priority order for worker groups. When multiple worker groups have the same highest score, the autoscaler picks the first one in the manifest. 

This is not always the desired behavior. For example, some worker groups might be added later (with `kubectl ray plugin`) to the cluster and we might want to prioritize the newer worker groups over the older ones. 

One other scenario we need to support is: we always prefer to use group 1 over group 2 if group 1 has resources available. But if group 1 is temporarily out of resources, ray cluster will fallback to group 2. With current implementation, once group 1 is out of resources, it will penalized later, and if group 2 never has failed scaling event, group 1 will never be picked again over group 2 even if group 1 has resources available later. 

## Proposed Solution

We would like introduce a new optional `priority` field to the worker group spec. The priority is a non-negative integer, and the higher the priority, the more preferred the worker group is. It will be used as a tie-breaker when multiple worker groups have the same highest score. The existing penalty logic of recent scaling up failures should still be there, but a worker group with higher priority should be have the penalty reduced to 0 or positive after a while to retry. 

## Detailed Design

To support relative priority order for worker groups, we will make changes across the configuration schema, the scheduler's node selection logic, and the cloud resource monitor's penalty mechanism.

### 1. Configuration Schema Changes

We will introduce an optional `priority` field to the worker group specification.

- **KubeRay CRD**: Add `priority` (int) to `workerGroupSpecs`.
- **Python Autoscaler Schema**:
    - Update `NodeTypeConfig` in `python/ray/autoscaler/v2/instance_manager/config.py` to include a `priority: int = 0` field.
    - Update `SchedulingNode` in `python/ray/autoscaler/v2/scheduler.py` to store the `priority` of its node type.

### 2. Priority-Aware Node Selection

The `ResourceDemandScheduler` will use a 4-level sorting key to select the "best" node. This ensures that priority acts as a deterministic tie-breaker after recovery status is accounted for.

In `python/ray/autoscaler/v2/scheduler.py`, the `_sched_best_node` method will be updated:

```python
# Sort the results by a multi-level key (higher is better for all levels):
results = sorted(
    results,
    key=lambda r: (
        r.score,                                              # 1. Utilization score
        recoverable_availabilities.get(r.node.node_type, 1.0), # 2. Common recovery status (0.0 to 1.0)
        r.node.priority,                                      # 3. User-defined priority
        original_availabilities.get(r.node.node_type, 1.0),    # 4. Original recency-based score
    ),
    reverse=True,
)
```

**Selection Logic**:
1. **Utilization**: Pick the node that schedules the most requests.
2. **Recovery Status**: If tied, pick the node that has progressed further in its "recovery" from the last failure. All node types use the same recovery slope.
3. **Priority**: If recovery status is equal (e.g., both have fully recovered to 1.0, or both have never failed), pick the one with the higher user-defined priority.
4. **Historical Recency**: If priorities are also equal, use the original `get_resource_availabilities()` score. This ensures that the node that failed longest ago (or never) is preferred.

### 3. Penalty Recovery

We will enhance `CloudResourceMonitor` by adding a new `get_recoverable_resource_availabilities` function. This function uses a uniform recovery slope for all node types.

- **Recovery Window**: `RAY_AUTOSCALER_AVAILABILITY_RECOVERY_S` (default: 60s).
- **Safety Floor**: `RAY_AUTOSCALER_MIN_RETRY_INTERVAL_S` (default: 5s).
- **Score**: `0.0` if `t < Floor`, else `min(1.0, t / Window)`.

```python
def get_recoverable_resource_availabilities(self) -> Dict[NodeType, float]:
    now = time.time()
    scores = {}
    for node_type, last_fail in self._last_unavailable_timestamp.items():
        t = now - last_fail
        if t < MIN_RETRY_INTERVAL:
            scores[node_type] = 0.0
        else:
            # Uniform recovery slope for all groups
            scores[node_type] = min(1.0, t / RECOVERY_WINDOW)
    return scores
```

### 4. Implementation Steps

1. **Update `NodeTypeConfig`**: Add `priority: int = 0` field.
2. **Modify `SchedulingNode`**: Store `priority` from its node type.
3. **Enhance `CloudResourceMonitor`**: Add `get_recoverable_resource_availabilities` with a uniform slope.
4. **Update `ResourceDemandScheduler`**: Implement the 4-level sorting key in `_sched_best_node`, utilizing both the new recovery score and the original recency score.
5. **KubeRay Integration**: Expose `priority` in the `RayCluster` CRD and ensure it is propagated to the autoscaler.

#### Existing Function: `get_resource_availabilities` (Unchanged)
This remains as the historical tie-breaker, ensuring that more recent failures always have a lower relative score than older ones, even after both have fully "recovered" in the slope-based model.
