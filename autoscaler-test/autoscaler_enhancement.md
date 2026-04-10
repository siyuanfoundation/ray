# Enhance Ray Autoscaler to Set Relative Priority Order for Worker Groups

## Problem Statement

Currently, Ray Autoscaler v2 does not support setting relative priority order for worker groups. When multiple worker groups have the same highest score, the autoscaler picks the first one in the manifest. 

This is not always the desired behavior. For example, some worker groups might be added later (with `kubectl ray plugin`) to the cluster and we might want to prioritize the newer worker groups over the older ones. 

One other scenario we need to support is: we always prefer to use group 1 over group 2 if group 1 has resources available. But if group 1 is temporarily out of resources, ray cluster will fallback to group 2. With current implementation, once group 1 is out of resources, it will penalized later, and if group 2 never has failed scaling event, group 1 will never be picked again over group 2 even if group 1 has resources available later. 

## Proposed Solution

We would like introduce a new optional `priority` field to the worker group spec. The priority is a non-negative integer, and the higher the priority, the more preferred the worker group is. It will be used as a tie-breaker when multiple worker groups have the same highest score. The existing penalty logic of recent scaling up failures should still be there, but a worker group with higher priority should be have the penalty reduced to 0 or positive after a while to retry. 

## Detailed Design

