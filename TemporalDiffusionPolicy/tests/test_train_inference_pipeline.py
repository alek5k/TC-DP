"""Test full training and inference pipeline"""
import pytest
from conftest import run_pipeline_test


class TestTrainInferencePipeline:
    """Test full training and inference pipeline"""

    def test_WaitAtGoal_Base(self):
        dataset_config = {"dataset_path": "data/demonstration/wait_at_goal_dataset_v4.zarr", "env_name": "WaitAtGoal"}
        encoding_config = {"name": "Base", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_WaitAtGoal_ID(self):
        dataset_config = {"dataset_path": "data/demonstration/wait_at_goal_dataset_v4.zarr", "env_name": "WaitAtGoal"}
        encoding_config = {"name": "ID", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_WaitAtGoal_SaP(self):
        dataset_config = {"dataset_path": "data/demonstration/wait_at_goal_dataset_v4.zarr", "env_name": "WaitAtGoal"}
        encoding_config = {"name": "SaP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        run_pipeline_test(dataset_config, encoding_config)

    def test_WaitAtGoal_SiP(self):
        dataset_config = {"dataset_path": "data/demonstration/wait_at_goal_dataset_v4.zarr", "env_name": "WaitAtGoal"}
        encoding_config = {"name": "SiP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_WaitAtGoal_ID_SaP(self):
        dataset_config = {"dataset_path": "data/demonstration/wait_at_goal_dataset_v4.zarr", "env_name": "WaitAtGoal"}
        encoding_config = {"name": "ID_SaP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        run_pipeline_test(dataset_config, encoding_config)

    def test_WaitAtGoal_ID_SiP(self):
        dataset_config = {"dataset_path": "data/demonstration/wait_at_goal_dataset_v4.zarr", "env_name": "WaitAtGoal"}
        encoding_config = {"name": "ID_SiP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_WaitAtGoal_PIP(self):
        dataset_config = {"dataset_path": "data/demonstration/wait_at_goal_dataset_v4.zarr", "env_name": "WaitAtGoal"}
        encoding_config = {"name": "PIP", "use_pause_induced_perturbation": True, "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_LiftQA_Base(self):
        dataset_config = {"dataset_path": "data/demonstration/lift_qa_v2.zarr", "env_name": "LiftQA"}
        encoding_config = {"name": "Base", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_LiftQA_ID(self):
        dataset_config = {"dataset_path": "data/demonstration/lift_qa_v2.zarr", "env_name": "LiftQA"}
        encoding_config = {"name": "ID", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_LiftQA_SaP(self):
        dataset_config = {"dataset_path": "data/demonstration/lift_qa_v2.zarr", "env_name": "LiftQA"}
        encoding_config = {"name": "SaP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        run_pipeline_test(dataset_config, encoding_config)

    def test_LiftQA_SiP(self):
        dataset_config = {"dataset_path": "data/demonstration/lift_qa_v2.zarr", "env_name": "LiftQA"}
        encoding_config = {"name": "SiP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_LiftQA_ID_SaP(self):
        dataset_config = {"dataset_path": "data/demonstration/lift_qa_v2.zarr", "env_name": "LiftQA"}
        encoding_config = {"name": "ID_SaP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        run_pipeline_test(dataset_config, encoding_config)

    def test_LiftQA_ID_SiP(self):
        dataset_config = {"dataset_path": "data/demonstration/lift_qa_v2.zarr", "env_name": "LiftQA"}
        encoding_config = {"name": "ID_SiP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)

    def test_LiftQA_PIP(self):
        dataset_config = {"dataset_path": "data/demonstration/lift_qa_v2.zarr", "env_name": "LiftQA"}
        encoding_config = {"name": "PIP", "use_pause_induced_perturbation": True, "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        run_pipeline_test(dataset_config, encoding_config)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

