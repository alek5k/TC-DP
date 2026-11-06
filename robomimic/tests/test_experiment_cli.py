"""Tests for experiment-CLI discovery labels."""
import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path


CLI_PATH = Path(__file__).resolve().parents[1] / "robomimic" / "scripts" / "experiment_cli.py"
SPEC = importlib.util.spec_from_file_location("robomimic_experiment_cli", CLI_PATH)
experiment_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = experiment_cli
SPEC.loader.exec_module(experiment_cli)


class TestExperimentCli(unittest.TestCase):

    def test_screen_session_name_is_short_and_unique(self):
        session = experiment_cli.make_screen_session(
            "train-lift-tc_diffusion_policy_mod-idleness_and_sinusoidal_progress_conditioning-seed1000"
        )
        next_session = experiment_cli.make_screen_session(
            "train-lift-tc_diffusion_policy_mod-idleness_and_sinusoidal_progress_conditioning-seed1001"
        )
        self.assertLessEqual(len(session), 64)
        self.assertTrue(session.startswith("robomimic-train-lift-"))
        self.assertNotEqual(session, next_session)

    def test_multi_seed_screen_batch_runs_direct_commands_in_order(self):
        experiment = experiment_cli.Experiment(
            "liftqa",
            "tc_diffusion_policy_mod_nocrop",
            (
                Path(__file__).resolve().parents[1]
                / "robomimic/exps/temporaldp/temporal/liftqa/tc_diffusion_policy_mod_nocrop.json"
            ),
        )
        with (
            mock.patch.object(experiment_cli, "select_experiment", return_value=experiment),
            mock.patch.object(experiment_cli, "prompt_gpu", return_value="1"),
            mock.patch.object(experiment_cli, "prompt_seeds", return_value=["1", "2", "3"]),
            mock.patch.object(experiment_cli, "prompt_execution_mode", return_value=experiment_cli.ExecutionMode(True)),
            mock.patch.object(experiment_cli, "prompt_yes_no", return_value=True),
            mock.patch.object(experiment_cli, "make_screen_session", return_value="batch"),
            mock.patch.object(experiment_cli, "start_screen_session", return_value=0) as start,
        ):
            experiment_cli.run_training([experiment], dry_run=False)

        command = start.call_args.args[1]
        self.assertNotIn("run-training-plan", command)
        self.assertLess(command.index("--seed 1"), command.index("--seed 2"))
        self.assertLess(command.index("--seed 2"), command.index("--seed 3"))
        self.assertIn("vla_alert", command)
        self.assertIn('exit "$status"', command)


if __name__ == "__main__":
    unittest.main()
