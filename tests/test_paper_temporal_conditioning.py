"""Fixed temporal conditioning remains usable in the paper branch."""
import numpy as np
import pytest
import zarr

from temporal_diffusion_policy.dataset_generation import NormalizationType, TemporalConditioningDataset
from temporal_diffusion_policy.env_mappings import get_env_settings_from_env_name


@pytest.mark.parametrize('progress', ['sinusoidal', 'saturating'])
def test_fixed_temporal_features_reset_at_episode_boundaries(tmp_path, progress):
    store = zarr.open(str(tmp_path / 'episodes.zarr'), mode='w')
    data, meta = store.create_group('data'), store.create_group('meta')
    data.array('action', np.zeros((8, 2), dtype=np.float32))
    data.array('agent_pose', np.zeros((8, 2), dtype=np.float32))
    data.array('agent_velocity', np.tile([0., 0., 1., 1.], 2).astype(np.float32))
    data.array('step_count', np.tile(np.arange(4), 2))
    data.array('full_image', np.zeros((8, 3, 4, 4), dtype=np.float32))
    meta.array('episode_ends', np.array([4, 8]))
    dataset = TemporalConditioningDataset(
        dataset_path=str(tmp_path / 'episodes.zarr'), pred_horizon=4,
        obs_horizon=2, action_horizon=2,
        dataset_keys_and_normalisation={'action': NormalizationType.MIN_MAX},
        use_idleness_encoding=True,
        use_sinusoidal_progress_encoding=progress == 'sinusoidal',
        sinusoidal_progress_encoding_parameters={
            'd_model': 8, 'power_value': 10000., 'normalize_by_episode_length': False,
        },
        use_saturating_progress_encoding=progress == 'saturating',
    )
    feature = progress + '_progress_encoding'
    for key in ['idleness', feature]:
        values = dataset.full_train_data[key]
        assert np.isfinite(values).all()
        np.testing.assert_allclose(values[:4], values[4:])
        assert dataset[0][key].shape[0] == 2
    assert dataset[0]['action'].shape == (4, 2)


@pytest.mark.parametrize('environment', ['PianoSim', 'orcahand', 'PushT', 'PushT-HighDim', 'PushT2'])
def test_paper_environment_registry_rejects_removed_environment(environment):
    with pytest.raises(ValueError):
        get_env_settings_from_env_name(environment)
