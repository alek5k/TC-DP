"""Shared pytest configuration and helper functions for all tests"""
import os
import shutil
import sys
import pytest
from pathlib import Path
import torch
import gc
import time

# Set environment variable for deterministic CUDA operations
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if package_path not in sys.path:
    sys.path.append(package_path)

# Import the main functions and configs
from temporal_diffusion_policy.training import main as training_main, ExperimentConfig
from temporal_diffusion_policy.inference import main as inference_main, InferenceConfig


TESTING_ROOT = "testing"
INFERENCE_ROOT = f"{TESTING_ROOT}/data/inference"
CHECKPOINT_ROOT = f"{TESTING_ROOT}/data/checkpoints"


def setup_testing_dir():
    """Create checkpoint directory if it doesn't exist"""
    os.makedirs(TESTING_ROOT, exist_ok=True)


def clean_testing_dir():
    """Delete all checkpoints in testing directory"""
    if os.path.exists(TESTING_ROOT):
        shutil.rmtree(TESTING_ROOT)
    setup_testing_dir()


def get_checkpoint_name(dataset_config, encoding_config):
    """Generate checkpoint name based on dataset and encoding config"""
    dataset_name = Path(dataset_config["dataset_path"]).stem
    env_name = dataset_config["env_name"]
    encoding_name = encoding_config["name"]
    return f"{env_name}_{dataset_name}_{encoding_name}"


def run_training_test(dataset_config, encoding_config):
    """Run training phase and return checkpoint path"""
    dataset_path = dataset_config["dataset_path"]
    dataset_name = Path(dataset_path).stem
    
    # Skip if dataset doesn't exist
    if not os.path.exists(dataset_path):
        pytest.skip(f"Dataset {dataset_path} not found")
    
    # ========== TRAINING PHASE ==========
    print(f"\n{'='*80}")
    print(f"TRAINING: {dataset_name} with {encoding_config['name']}")
    print(f"{'='*80}")
    
    # Generate checkpoint name
    checkpoint_name = get_checkpoint_name(dataset_config, encoding_config)
    checkpoint_path = os.path.join(CHECKPOINT_ROOT, checkpoint_name)
    
    # Build training config
    train_config = ExperimentConfig(
        dataset_path=dataset_path,
        checkpoint_root=CHECKPOINT_ROOT,
        override_checkpoint_name=checkpoint_name,
        batch_size=256,
        num_epochs=1,
        num_dataloader_workers=1,
        use_wandb=False,
        use_idleness_encoding=encoding_config['use_idleness_encoding'],
        use_sinusoidal_progress_encoding=encoding_config['use_sinusoidal_progress_encoding'],
        use_saturating_progress_encoding=encoding_config['use_saturating_progress_encoding'],
    )
    
    # Run training
    try:
        training_main(train_config)
    except Exception as e:
        pytest.fail(f"Training failed for {dataset_name} with {encoding_config['name']}: {str(e)}")
    
    # Verify checkpoint directory was created
    assert os.path.exists(checkpoint_path), f"Checkpoint directory not created: {checkpoint_path}"
    
    # Verify final.pth exists
    final_pth = Path(checkpoint_path) / "final.pth"
    assert final_pth.exists(), f"final.pth not found in {checkpoint_path}"
    
    print(f"✓ Training completed: {checkpoint_path}")
    
    return checkpoint_path


def run_inference_test(env_name, encoding_config, checkpoint_path):
    """Run inference phase using provided checkpoint"""
    
    # ========== INFERENCE PHASE ==========
    print(f"\n{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    # Generate unique suffix for inference dataset to avoid stdin prompts
    unique_suffix = f"test_{int(time.time() * 1000)}"
    
    # Build inference config using the checkpoint we just created
    inference_config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        env_name=env_name,
        num_episodes=2,
        save_video=False,
        render=False,
        max_steps=30,
        inference_dataset_root=INFERENCE_ROOT,
        generated_dataset_suffix=unique_suffix,
        use_idleness_encoding=encoding_config['use_idleness_encoding'],
        use_sinusoidal_progress_encoding=encoding_config['use_sinusoidal_progress_encoding'],
        use_saturating_progress_encoding=encoding_config['use_saturating_progress_encoding'],
        sinusoidal_progress_encoding_dim=encoding_config.get('sinusoidal_progress_encoding_dim', 4)
    )
    
    # Run inference
    try:
        inference_main(inference_config)
    except Exception as e:
        pytest.fail(f"Inference failed for {env_name} with {encoding_config['name']}: {str(e)}")
    
    print(f"✓ Inference completed for {env_name} with {encoding_config['name']}")


def run_pipeline_test(dataset_config, encoding_config):
    """Shared test logic for training and inference pipeline"""
    # Run training
    checkpoint_path = run_training_test(dataset_config, encoding_config)
    env_name = dataset_config["env_name"]
    # Run inference
    run_inference_test(env_name, encoding_config, checkpoint_path)
    
    print(f"✓ Full pipeline completed for {dataset_config['env_name']} with {encoding_config['name']}")
    
    # Clear GPU memory after each test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def setup_teardown():
    """Setup and teardown for all tests"""
    clean_testing_dir()
    yield
    # Teardown: optionally clean up after all tests
    clean_testing_dir()
