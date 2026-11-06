"""Test inference using existing checkpoints to validate trained models"""
import pytest
from conftest import run_inference_test


class TestInferenceExistingCheckpoints:
    """Test inference using existing datasets to validate checkpoints"""

    def test_WaitAtGoal_Base_existing(self):
        """Test WaitAtGoal with Base encoding using existing checkpoint"""
        encoding_config = {"name": "Base", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        checkpoint_path = "data/checkpoints/WaitAtGoal_20260114_180054"
        run_inference_test("WaitAtGoal", encoding_config, checkpoint_path)

    def test_WaitAtGoal_ID_existing(self):
        """Test WaitAtGoal with ID encoding using existing checkpoint"""
        encoding_config = {"name": "ID", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        checkpoint_path = "data/checkpoints/WaitAtGoal_20260114_180104"
        run_inference_test("WaitAtGoal", encoding_config, checkpoint_path)

    def test_WaitAtGoal_SaP_existing(self):
        """Test WaitAtGoal with SaP encoding using existing checkpoint"""
        encoding_config = {"name": "SaP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        checkpoint_path = "data/checkpoints/WaitAtGoal_20260121_140503"
        run_inference_test("WaitAtGoal", encoding_config, checkpoint_path)

    def test_WaitAtGoal_SiP_existing(self):
        """Test WaitAtGoal with SiP encoding using existing checkpoint"""
        encoding_config = {"name": "SiP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False, "sinusoidal_progress_encoding_dim": 32}
        checkpoint_path = "data/checkpoints/WaitAtGoal_20260115_045503"
        run_inference_test("WaitAtGoal", encoding_config, checkpoint_path)

    def test_WaitAtGoal_ID_SaP_existing(self):
        """Test WaitAtGoal with ID_SaP encoding using existing checkpoint"""
        encoding_config = {"name": "ID_SaP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        checkpoint_path = "data/checkpoints/WaitAtGoal_20260114_193114"
        run_inference_test("WaitAtGoal", encoding_config, checkpoint_path)

    def test_WaitAtGoal_ID_SiP_existing(self):
        """Test WaitAtGoal with ID_SiP encoding using existing checkpoint"""
        encoding_config = {"name": "ID_SiP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False, "sinusoidal_progress_encoding_dim": 32}
        checkpoint_path = "data/checkpoints/WaitAtGoal_20260115_183507"
        run_inference_test("WaitAtGoal", encoding_config, checkpoint_path)

    def test_WaitAtGoal_PIP_existing(self):
        """Test WaitAtGoal with PIP encoding using existing checkpoint"""
        encoding_config = {"name": "PIP", "use_pause_induced_perturbation": True, "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        checkpoint_path = "data/checkpoints/WaitAtGoal_20260114_180054"
        run_inference_test("WaitAtGoal", encoding_config, checkpoint_path)

    def test_LiftQA_Base_existing(self):
        """Test LiftQA with Base encoding using existing checkpoint"""
        encoding_config = {"name": "Base", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        checkpoint_path = "data/checkpoints/LiftQA_20260122_000453"
        run_inference_test("LiftQA", encoding_config, checkpoint_path)

    def test_LiftQA_ID_existing(self):
        """Test LiftQA with ID encoding using existing checkpoint"""
        encoding_config = {"name": "ID", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        checkpoint_path = "data/checkpoints/LiftQA_20260122_000503"
        run_inference_test("LiftQA", encoding_config, checkpoint_path)

    def test_LiftQA_SaP_existing(self):
        """Test LiftQA with SaP encoding using existing checkpoint"""
        encoding_config = {"name": "SaP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        checkpoint_path = "data/checkpoints/LiftQA_20260122_015518"
        run_inference_test("LiftQA", encoding_config, checkpoint_path)

    def test_LiftQA_SiP_existing(self):
        """Test LiftQA with SiP encoding using existing checkpoint"""
        encoding_config = {"name": "SiP", "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False, "sinusoidal_progress_encoding_dim": 256}
        checkpoint_path = "data/checkpoints/LiftQA_20260122_123617"
        run_inference_test("LiftQA", encoding_config, checkpoint_path)

    def test_LiftQA_ID_SaP_existing(self):
        """Test LiftQA with ID_SaP encoding using existing checkpoint"""
        encoding_config = {"name": "ID_SaP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": True}
        checkpoint_path = "data/checkpoints/LiftQA_20260122_034232"
        run_inference_test("LiftQA", encoding_config, checkpoint_path)

    def test_LiftQA_ID_SiP_existing(self):
        """Test LiftQA with ID_SiP encoding using existing checkpoint"""
        encoding_config = {"name": "ID_SiP", "use_idleness_encoding": True, "use_sinusoidal_progress_encoding": True, "use_saturating_progress_encoding": False, "sinusoidal_progress_encoding_dim": 256}
        checkpoint_path = "data/checkpoints/LiftQA_20260122_090200"
        run_inference_test("LiftQA", encoding_config, checkpoint_path)

    def test_LiftQA_PIP_existing(self):
        """Test LiftQA with PIP encoding using existing checkpoint"""
        encoding_config = {"name": "PIP", "use_pause_induced_perturbation": True, "use_idleness_encoding": False, "use_sinusoidal_progress_encoding": False, "use_saturating_progress_encoding": False}
        checkpoint_path = "data/checkpoints/LiftQA_20260122_000453"
        run_inference_test("LiftQA", encoding_config, checkpoint_path)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
