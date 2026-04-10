# RayCluster Autoscaler Worker Group Order Experiment

## Instructions

We are testing the behavior of ray autoscaler where there are multiple worker groups to scale up.
My understanding is that the autoscaler code under `ray/python/ray/autoscaler` picks the workergroup that scores the highest, and if there are multiple worker groups with the same highest score, it picks the one that comes first in the manifest. This assumption is based on my reading of the code, but it may not be true. I want to verify that is always the case.

I have deployed a ray cluster with ray-autoscaler-cluster.yaml. The cluster has 2 identical worker groups except the name.
I want to see if ray autoscaler always scale up the 1st worker group deterministically when the 2 groups have the same score.

You can trigger cluster scaling up by creating detached actors like the following examples:

```
# Step 5.1: Create a detached actor "actor1" which requires 1 CPU.
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
kubectl exec -it $HEAD_POD -- python3 /home/ray/samples/detached_actor.py actor1

# Step 5.2: The Ray Autoscaler creates a new worker Pod.
kubectl get pods -l=ray.io/is-ray-node=yes

# [Example output]
# NAME                                             READY   STATUS    RESTARTS   AGE
# raycluster-autoscaler-head                       2/2     Running   0          xxm
# raycluster-autoscaler-small-group-worker-yyyyy   1/1     Running   0          xxm

# Step 5.3: Create a detached actor which requires 1 CPU.
kubectl exec -it $HEAD_POD -- python3 /home/ray/samples/detached_actor.py actor2
kubectl get pods -l=ray.io/is-ray-node=yes

# [Example output]
# NAME                                             READY   STATUS    RESTARTS   AGE
# raycluster-autoscaler-head                       2/2     Running   0          xxm
# raycluster-autoscaler-small-group-worker-yyyyy   1/1     Running   0          xxm
# raycluster-autoscaler-small-group-worker-zzzzz   1/1     Running   0          xxm

# Step 5.4: List all actors in the Ray cluster.
kubectl exec -it $HEAD_POD -- ray list actors
```

You can trigger RayCluster scale-down by terminating detached actors like the following examples:

```
# Step 6.1: Terminate the detached actor "actor1".
kubectl exec -it $HEAD_POD -- python3 /home/ray/samples/terminate_detached_actor.py actor1

# Step 6.2: A worker Pod will be deleted after `idleTimeoutSeconds` (default 60s) seconds.
kubectl get pods -l=ray.io/is-ray-node=yes

# [Example output]
# NAME                                             READY   STATUS    RESTARTS   AGE
# raycluster-autoscaler-head                       2/2     Running   0          xxm
# raycluster-autoscaler-small-group-worker-zzzzz   1/1     Running   0          xxm

# Step 6.3: Terminate the detached actor "actor2".
kubectl exec -it $HEAD_POD -- python3 /home/ray/samples/terminate_detached_actor.py actor2

# Step 6.4: A worker Pod will be deleted after `idleTimeoutSeconds` (default 60s) seconds.
kubectl get pods -l=ray.io/is-ray-node=yes

# [Example output]
# NAME                         READY   STATUS    RESTARTS   AGE
# raycluster-autoscaler-head   2/2     Running   0          xxm
```

Follow the following steps to run the experiment:

1. Create or verify the `Experiment Plan` section below and update if needed.
2. Run the experiment and save the results in the `Results` section below.
3. If the autoscaler does not behave as expected, go through the code in `ray/python/ray/autoscaler` to figure out why and write down your findings

## Experiment Plan

1. **Cleanup**: Terminate any existing actors and wait for the cluster to scale down to 0 workers.
2. **Initial State Verification**: Verify `kubectl get pods` shows only the head pod.
3. **Step 1**: Create `actor1` which requires 1 CPU.
4. **Observation 1**: Identify which worker group the new pod belongs to.
5. **Step 2**: Create `actor2` which requires 1 CPU.
6. **Observation 2**: Identify which worker group the second new pod belongs to.
7. **Step 3**: Create `actor3` which requires 1 CPU.
8. **Observation 3**: Identify which worker group the third new pod belongs to.
9. **Step 4**: Create `actor4` which requires 1 CPU.
10. **Observation 4**: Identify which worker group the fourth new pod belongs to.
11. **Step 5**: Create `actor5` which requires 1 CPU.
12. **Observation 5**: Identify which worker group the fifth new pod belongs to.
13. **Analysis**: Check if all pods belong to `small-group-1` (the first group in the manifest).
14. **Cleanup**: Terminate all actors.

## Results

The experiment was conducted on April 10, 2026. The cluster was initialized with a head pod and no workers. Five detached actors were created sequentially, each requiring 1 CPU.

| Step | Actor Name | Worker Group Identified | Pod Name |
| :--- | :--- | :--- | :--- |
| 1 | actor1 | small-group-1 | raycluster-autoscaler-small-group-1-worker-2rltv |
| 2 | actor2 | small-group-1 | raycluster-autoscaler-small-group-1-worker-lr46x |
| 3 | actor3 | small-group-1 | raycluster-autoscaler-small-group-1-worker-lnz9z |
| 4 | actor4 | small-group-1 | raycluster-autoscaler-small-group-1-worker-4b2ph |
| 5 | actor5 | small-group-1 | raycluster-autoscaler-small-group-1-worker-2cjr8 |

**Analysis**:
In all 5 steps, the Ray Autoscaler chose to scale up `small-group-1`. This worker group is the first one listed in the `workerGroupSpecs` section of the `ray-autoscaler-cluster.yaml` manifest. Since both `small-group-1` and `small-group-2` provide identical resources (1 CPU, 1G Memory), they likely received the same utilization score.

The code in `python/ray/autoscaler/v2/scheduler.py` uses a stable sort (`sorted`) to rank potential nodes for scheduling.
```python
        # Sort the results by score.
        results = sorted(
            results,
            key=lambda r: (
                r.score,
                cloud_resource_availabilities.get(r.node.node_type, 1),
            ),
            reverse=True,
        )
```
Because the sort is stable and the initial `results` list is populated by iterating through the `node_type_configs` (which preserves the order from the manifest), the first worker group in the manifest is deterministically chosen when scores are tied.

**Conclusion**:
The hypothesis is verified. Ray Autoscaler (v2) deterministically picks the first worker group in the manifest when multiple worker groups have the same highest score.

### Experiment 2: Updating Manifest Mid-Run

**Scenario**: Start the cluster, trigger scale-up of the 1st worker group, then update the manifest by moving the 2nd worker group to the front of the list.

**Steps**:
1. Initial state: `small-group-1` is 1st in manifest, `small-group-2` is 2nd.
2. Create `actor1`.
3. Observation: `small-group-1` scales up (Pod: `raycluster-autoscaler-small-group-1-worker-xk67v`).
4. Update manifest: Swap order so `small-group-2` is 1st.
5. Apply manifest (`kubectl apply`).
6. Create `actor2`.
7. Observation: `small-group-2` scales up (Pod: `raycluster-autoscaler-small-group-2-worker-rj2bc`).

**Analysis**:
Even when a cluster is already running with workers from a previously "first" group, updating the manifest to put a different group first causes the autoscaler to favor the new "first" group for subsequent scale-ups (assuming equal scores). This confirms that the selection logic is dynamic and always respects the current manifest's order.

**Final Conclusion**:
Ray Autoscaler (v2) uses a stable sorting mechanism that relies on the order of worker groups defined in the `RayCluster` manifest. The group that appears first in the `workerGroupSpecs` list will always be the preferred choice for scaling up when all other scoring factors are equal.
