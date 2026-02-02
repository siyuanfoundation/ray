import logging
import shutil
from typing import Any, Dict, Optional, Union

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import args as ocp_args
from orbax.checkpoint import type_handlers

from ray.train import Checkpoint, CheckpointConfig
from ray.train._internal.checkpoint_manager import _CheckpointManager
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import StorageContext, _delete_fs_path, _exists_at_fs_path

logger = logging.getLogger(__name__)


class JaxCheckpointManager(_CheckpointManager):
    """A CheckpointManager that wraps orbax.CheckpointManager and handles communication with Ray.

    This class manages the lifecycle of checkpoints using both Orbax for storage
    and Ray Train's _CheckpointManager for decision making (keeping top K checkpoints
    based on metrics).

    Args:
        storage_context: The storage context for the checkpoint.
        checkpoint_config: The Ray Train CheckpointConfig to control checkpoint
            keeping/deletion.
    """

    def __init__(
        self,
        storage_context: StorageContext,
        checkpoint_config: Optional[CheckpointConfig] = None,
        enable_async_checkpointing: bool = True,
    ):
        super().__init__(checkpoint_config=checkpoint_config)
        self._storage_context = storage_context

        checkpoint_config = checkpoint_config or CheckpointConfig()

        options = ocp.CheckpointManagerOptions(
            max_to_keep=checkpoint_config.num_to_keep,
            best_mode=checkpoint_config.checkpoint_score_order,
            enable_async_checkpointing=enable_async_checkpointing,
        )   
        if checkpoint_config.checkpoint_score_attribute: 
            options.best_fn = lambda metrics: metrics[checkpoint_config.checkpoint_score_attribute]
        if jax.process_index() == 0:
            self._cleanup_if_exists()

        # Ensure the directory exists
        # StorageContext.experiment_fs_path already includes the experiment_dir_name
        self._storage_context.storage_filesystem.create_dir(
            self._storage_context.experiment_fs_path
        )
        
        # Use PyTreeCheckpointHandler for standard PyTree saving
        item_handlers = {
            "items": ocp.PyTreeCheckpointHandler(use_ocdbt=True, use_zarr3=True)
        }
        
        self._orbax_manager = ocp.CheckpointManager(
            directory=self._storage_context.experiment_fs_path,
            item_handlers=item_handlers,
            options=options,
        )

    def _cleanup_if_exists(self):
        fs = self._storage_context.storage_filesystem
        path = self._storage_context.experiment_fs_path
        
        if _exists_at_fs_path(fs, path):
            logger.info(f"Cleaning up existing checkpoint directory: {path}")
            try:
                _delete_fs_path(fs, path)
            except Exception as e:
                logger.warning(f"Error during checkpoint directory cleanup: {e}")

    def wait_until_finished(self):
        self._orbax_manager.wait_until_finished()

    def save(
        self,
        step: int, 
        train_state: dict, 
        metrics: Optional[Dict] = None,
        force: bool = False,
        chunk_byte_size=1024*1024*1024,
    ) -> str:
        """Saves a checkpoint using Orbax and registers it with Ray's CheckpointManager.

        Args:
            step: The training step (used as checkpoint ID).
            train_state: The train state to save.
            metrics: Metrics associated with this checkpoint, used for top-k keeping.
            force: Whether to force save even if not scheduled (Orbax logic).
            chunk_byte_size: The chunk size for the save operation, default is 1GB.

        Returns:
            The path to the saved checkpoint.
        """

        # Prepare save arguments
        # ocdbt_target_data_file_size controls the chunk size (1GB here)
        save_args = ocp_args.PyTreeSave(
            item=train_state,
            save_args=jax.tree.map(
                lambda _: ocp.SaveArgs(chunk_byte_size=chunk_byte_size), 
                train_state
            )
        )
        success = self._orbax_manager.save(
            step, 
            args=ocp_args.Composite(items=save_args),
            force=force, 
            metrics=metrics if metrics else {},
        )
        
        if success or success is None:            
            # Register with Ray
            checkpoint_path = f"{self._storage_context.experiment_fs_path}/{step}"
            
            # Helper to create Checkpoint matching the storage context filesystem
            checkpoint = Checkpoint(
                filesystem=self._storage_context.storage_filesystem,
                path=checkpoint_path,
            )
            
            self.register_checkpoint(
                _TrainingResult(checkpoint=checkpoint, metrics=metrics or {})
            )
            
            logger.info(f"Saved and registered checkpoint for step {step}")
            return checkpoint_path
        return ""

    def restore(
        self,
        target: dict,
        step: Optional[int] = None, 
    ) -> dict:
        """
        Restores a distributed model checkpoint given the target abstract state with sharding.
        This method should be called by all hosts.
        
        Args:
            target: A PyTree with the desired structure and encodings (sharding).
            step: The step to restore.
        """
        # 1. Create RestoreArgs with Sharding info
        # This maps every leaf in the tree to an ArrayRestoreArgs object containing its sharding
        # This is CRITICAL for restoring into a distributed mesh correctly
        def map_to_restore_args(leaf):
            if hasattr(leaf, 'sharding'):
                return type_handlers.ArrayRestoreArgs(sharding=leaf.sharding)
            return type_handlers.RestoreArgs()

        restore_args_structure = jax.tree.map(map_to_restore_args, target)

        # 2. Create the PyTreeRestore object
        # 'item' is the abstract structure (target)
        # 'restore_args' contains the distributed reading instructions
        checkpoint_args = ocp_args.PyTreeRestore(
            item=target,
            restore_args=restore_args_structure
        )

        # 3. Restore
        # This returns a dictionary because of the item options
        restored = self._orbax_manager.restore(
            step, 
            args=ocp_args.Composite(items=checkpoint_args)
        )
        
        # Extract the actual state
        logger.info(f"Restored checkpoint for step {step}")
        return restored['items']

    @property
    def orbax_manager(self) -> ocp.CheckpointManager:
        return self._orbax_manager

