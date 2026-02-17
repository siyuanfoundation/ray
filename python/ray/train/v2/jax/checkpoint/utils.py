import os
import tempfile
import uuid
from typing import Any, Dict, Optional

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import type_handlers

import ray.train
from ray.train import Checkpoint


def save_checkpoint(
    item: Any,
    path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Saves a checkpoint using Orbax PyTreeCheckpointer and reports it to Ray Train.

    Args:
        item: The PyTree to save.
        path: Optional path to save the checkpoint to. If not provided, a temporary directory is used.
        metrics: Optional dictionary of metrics to report to Ray Train along with the checkpoint.
    """
    if path:
        checkpoint_dir = path
    else:
        checkpoint_dir = os.path.join(
            tempfile.gettempdir(), f"checkpoint_{uuid.uuid4()}"
        )

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(checkpoint_dir, item)

    # Report to Ray Train
    ray.train.report(
        metrics or {},
        checkpoint=Checkpoint.from_directory(checkpoint_dir),
    )


def restore_checkpoint(
    target: Any,
) -> Optional[Any]:
    """
    Restores the latest checkpoint reported to Ray Train.

    Args:
        target: The target PyTree structure (with sharding info) to restore into.
                Restoration will enforce the sharding found in this target.

    Returns:
        The restored PyTree, or None if no checkpoint exists.
    """
    checkpoint = ray.train.get_checkpoint()
    if not checkpoint:
        return None

    with checkpoint.as_directory() as checkpoint_dir:
        # Infer restore_args from target to enforce sharding
        restore_args = jax.tree_util.tree_map(
            lambda x: type_handlers.ArrayRestoreArgs(
                mesh=x.sharding.mesh, sharding=x.sharding
            )
            if isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
            and hasattr(x, "sharding")
            else ocp.checkpoint_utils.construct_restore_args(x),
            target,
            is_leaf=lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct)),
        )
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(
            checkpoint_dir, item=target, restore_args=restore_args
        )
        return restored
