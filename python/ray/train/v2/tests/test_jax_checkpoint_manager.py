
import pytest
from unittest.mock import MagicMock, patch
import shutil
import sys
import os

# Skip tests if orbax is not available for import
ocp = pytest.importorskip("orbax.checkpoint")

import jax
from ray.train import CheckpointConfig
from ray.train._internal.storage import StorageContext
from ray.train.v2.jax.checkpoint import JaxCheckpointManager

# python -m pytest -v -s python/ray/train/v2/tests/test_jax_checkpoint_manager.py

@pytest.fixture
def storage_context(tmp_path):
    storage_path = tmp_path / "storage"
    return StorageContext(
        storage_path=str(storage_path),
        experiment_dir_name="test_experiment",
    )


def test_init_creates_orbax_manager(storage_context):
    with patch("orbax.checkpoint.CheckpointManager") as mock_orbax_cls:
        manager = JaxCheckpointManager(storage_context=storage_context)
        
        # Verify Orbax Manager was initialized with the correct path
        mock_orbax_cls.assert_called_once()
        call_args = mock_orbax_cls.call_args
        # In current implementation, options is passed if None
        assert "options" in call_args.kwargs
        assert call_args.kwargs["options"].max_to_keep is None


def test_save_calls_orbax_and_registers(storage_context):
    with patch("orbax.checkpoint.CheckpointManager") as mock_orbax_cls:
        # mock successful save
        mock_orbax_instance = mock_orbax_cls.return_value
        mock_orbax_instance.save.return_value = True
        
        manager = JaxCheckpointManager(storage_context=storage_context)
        
        step = 10
        train_state = {"params": {"w": 1, "b": 0}}
        metrics = {"accuracy": 0.9}
        
        path = manager.save(step=step, train_state=train_state, metrics=metrics)
        
        # Verify Orbax save was called
        mock_orbax_instance.save.assert_called_once()
        _, kwargs = mock_orbax_instance.save.call_args
        assert kwargs["metrics"] == metrics
        
        # Verify Checkpoint was registered with Ray
        assert manager.latest_checkpoint_result is not None
        latest_result = manager.latest_checkpoint_result
        assert latest_result.metrics == metrics
        assert str(step) in latest_result.checkpoint.path


def test_save_failure_does_not_register(storage_context):
    with patch("orbax.checkpoint.CheckpointManager") as mock_orbax_cls:
        mock_orbax_instance = mock_orbax_cls.return_value
        mock_orbax_instance.save.return_value = False # Simulate failure
        
        manager = JaxCheckpointManager(storage_context=storage_context)
        
        step = 10
        train_state = {"params": {}}
        
        path = manager.save(step=step, train_state=train_state)
        
        # Should return empty string on failure
        assert path == ""
        # Should not register checkpoint
        assert manager.latest_checkpoint_result is None


def test_restore_calls_orbax(storage_context):
    with patch("orbax.checkpoint.CheckpointManager") as mock_orbax_cls:
        mock_orbax_instance = mock_orbax_cls.return_value
        expected_restored = {"params": [1, 2, 3]}
        mock_orbax_instance.restore.return_value = {"items": expected_restored}
        
        manager = JaxCheckpointManager(storage_context=storage_context)
        
        target = {"params": [0, 0, 0]}
        restored = manager.restore(step=5, target=target)
        
        assert restored == expected_restored
        mock_orbax_instance.restore.assert_called_once()
        # restore args: step, args=...
        # Check args
        args, kwargs = mock_orbax_instance.restore.call_args
        assert args[0] == 5


def test_cleanup_if_exists(storage_context):
    # Create the directory beforehand
    fs = storage_context.storage_filesystem
    path = storage_context.experiment_fs_path
    fs.create_dir(path)
    
    # Create a dummy file to verify deletion
    with fs.open_output_stream(f"{path}/dummy") as f:
        f.write(b"data")
    
    with patch("orbax.checkpoint.CheckpointManager"):
        with patch("jax.process_index", return_value=0):
            # This calls _cleanup_if_exists internally because jax.process_index() is 0
            manager = JaxCheckpointManager(storage_context=storage_context)
            
            # Verify mock called
            # (Implicitly verified if cleanup happens)
    
    # Check if directory was recreated (empty) or verified
    # The _cleanup_if_exists deletes it. Then init recreates it.
    # So the dummy file should be gone.
    file_info = fs.get_file_info(f"{path}/dummy")
    assert file_info.type == file_info.type.NotFound
    
    # Directory should exist (recreated)
    dir_info = fs.get_file_info(path)
    assert dir_info.type == dir_info.type.Directory


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
