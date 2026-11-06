"""Configuration-level checks for TemporalDiffusionPolicy baselines."""
import json
from pathlib import Path
import unittest

from robomimic.config import config_factory


CONFIG_ROOT = (
    Path(__file__).resolve().parents[1]
    / "robomimic"
    / "exps"
    / "temporaldp"
    / "temporal"
)

VARIANTS = {
    "bc": ("SpatialSoftmax", "CropRandomizer"),
    "bc_nocrop": ("SpatialSoftmax", None),
    "bc_rnn": ("SpatialSoftmax", "CropRandomizer"),
    "bc_mod": ("SpatialMeanPool", "CropRandomizer"),
    "bc_rnn_mod": ("SpatialMeanPool", "CropRandomizer"),
    "bc_mod_nocrop": ("SpatialMeanPool", None),
    "bc_rnn_mod_nocrop": ("SpatialMeanPool", None),
    "diffusion_policy_mod": ("SpatialMeanPool", "CropRandomizer"),
    "tc_diffusion_policy_mod": ("SpatialMeanPool", "CropRandomizer"),
    "diffusion_policy_mod_nocrop": ("SpatialMeanPool", None),
    "tc_diffusion_policy_mod_nocrop": ("SpatialMeanPool", None),
}


class TestTemporalConfigs(unittest.TestCase):
    def test_configs_load_with_expected_observations(self):
        config_paths = sorted(CONFIG_ROOT.glob("*/*.json"))
        self.assertEqual(len(config_paths), 22)
        for path in config_paths:
            with path.open() as handle:
                external = json.load(handle)
            config = config_factory(external["algo_name"])
            with config.values_unlocked():
                config.update(external)
            expected_obs_keys = ["agent_pose", "image"]
            if path.stem.startswith("tc_diffusion_policy_mod"):
                expected_obs_keys += ["idleness", "sinusoidal_progress_encoding"]
            self.assertEqual(config.all_obs_keys, sorted(expected_obs_keys), path)
            self.assertTrue(config.experiment.rollout.enabled, path)
            self.assertTrue(config.experiment.save.on_best_rollout_success_rate, path)
            self.assertEqual(config.train.data, f"datasets/temporal/{path.parent.name}_image.hdf5", path)
            self.assertEqual(config.train.action_config["actions"]["normalization"], "min_max", path)
            expected_pool, expected_randomizer = VARIANTS[path.stem]
            self.assertEqual(config.observation.encoder.rgb.core_kwargs.pool_class, expected_pool, path)
            self.assertEqual(config.observation.encoder.rgb.obs_randomizer_class, expected_randomizer, path)
            if "diffusion_policy" in path.stem:
                self.assertEqual(config.train.seq_length, 16, path)
                self.assertEqual(config.train.frame_stack, 2, path)


if __name__ == "__main__":
    unittest.main()
