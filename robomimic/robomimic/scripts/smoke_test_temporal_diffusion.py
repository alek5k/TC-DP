"""Run one in-memory training update and native rollout for a diffusion config.

This is intentionally smaller than ``train.py --debug``: it validates the
complete data, model, temporal-feature, action-normalization, and rollout
path without writing multi-gigabyte Diffusion Policy checkpoints.
"""
import argparse
import json
import os

from torch.utils.data import DataLoader

from robomimic.algo import RolloutPolicy, algo_factory
from robomimic.config import config_factory
from robomimic.utils import env_utils as EnvUtils
from robomimic.utils import file_utils as FileUtils
from robomimic.utils import obs_utils as ObsUtils
from robomimic.utils import torch_utils as TorchUtils
from robomimic.utils import train_utils as TrainUtils


def load_config(path):
    with open(path) as handle:
        external = json.load(handle)
    config = config_factory(external["algo_name"])
    with config.values_unlocked():
        config.update(external)
        config.train.data = [{"path": config.train.data}]
        config.train.num_epochs = 1
        for optim_params in config.algo.optim_params.values():
            optim_params["num_train_batches"] = 1
            optim_params["num_epochs"] = 1
    config.lock()
    return config


def smoke_test(config_path, inference_timesteps=10):
    config = load_config(config_path)
    if config.algo_name != "diffusion_policy":
        raise ValueError("This smoke runner supports diffusion_policy configs only")

    # A short DDIM rollout exercises the same policy path while keeping this
    # check practical on development hardware.
    with config.values_unlocked():
        config.algo.ddpm.enabled = False
        config.algo.ddim.enabled = True
        config.algo.ddim.num_inference_timesteps = inference_timesteps
    config.lock()

    ObsUtils.initialize_obs_utils_with_config(config)
    dataset_cfg = config.train.data[0]
    dataset_path = os.path.expanduser(dataset_cfg["path"])
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    temporal_cfg = config.observation["temporal_encodings"]
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config=dataset_cfg,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        temporal_cfg=temporal_cfg,
    )
    trainset, _ = TrainUtils.load_data_for_training(config, shape_meta["all_obs_keys"])
    action_normalization_stats = trainset.get_action_normalization_stats()
    batch = next(iter(DataLoader(trainset, batch_size=1, num_workers=0)))

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    batch = model.process_batch_for_training(batch)
    batch = model.postprocess_batch_for_training(batch, obs_normalization_stats=None)
    train_info = model.train_on_batch(batch=batch, epoch=1, validate=False)
    model.on_gradient_step()
    model.on_epoch_end(epoch=1)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=True,
    )
    env = EnvUtils.wrap_env_from_config(env=env, config=config)
    try:
        rollout_logs, _ = TrainUtils.rollout_with_stats(
            policy=RolloutPolicy(model, action_normalization_stats=action_normalization_stats),
            envs={env_meta["env_name"]: env},
            horizon=1,
            use_goals=False,
            num_episodes=1,
            render=False,
            terminate_on_success=True,
        )
    finally:
        env.close()

    print(
        "PASS: {} | loss={:.6f} | rollout={}".format(
            config_path,
            train_info["losses"]["l2_loss"].item(),
            rollout_logs,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    args = parser.parse_args()
    if args.inference_timesteps < 1:
        raise ValueError("--inference-timesteps must be positive")
    smoke_test(args.config, args.inference_timesteps)


if __name__ == "__main__":
    main()
