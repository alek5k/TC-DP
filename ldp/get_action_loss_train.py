import sys
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import numpy as np
import os
import pathlib
import click
import hydra
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import dill
from omegaconf import OmegaConf, open_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import einops

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def get_sampled(nactions):
    # Define the shape of nactions
    batch_size, timesteps, features = nactions.shape

    # Generate uniform random values between 0 and 2π along batch axis
    nactions = torch.rand(batch_size, 1, features) * 2 * torch.pi

    # Add 0.1 for each timestep
    timesteps_offset = torch.arange(timesteps, device=nactions.device).view(1, -1, 1) * 0.1
    nactions = nactions + timesteps_offset

    # Map each value to 2 * sin(value)
    nactions = 2 * torch.sin(nactions)

    return nactions

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-e', '--epochs', default=20, type=int)
@click.option('-t', '--transport', default=False, type=bool)
@click.option('--hidden_dim', default=64, type=int)
@click.option('--noprint', default=True, type=bool)
def main(checkpoint, output_dir, device, epochs, transport, hidden_dim, noprint):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    with open_dict(cfg):
        cfg.task.dataset.subsample_frames=1
        cfg.task.dataset.pad_before=20
        print(OmegaConf.to_yaml(cfg))

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload)

    policy = workspace.model
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    cfg.dataloader.batch_size = 256
    cfg.val_dataloader.batch_size = 256
    cfg.dataloader.num_workers = 4
    cfg.val_dataloader.num_workers = 4
    dataloader = DataLoader(dataset, **cfg.dataloader)

    val_dataset = dataset.get_validation_dataset()
    val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    policy.set_normalizer(normalizer)
    device = torch.device(device)
    policy.to(device)
    
    horizon = cfg.global_obs + 1
    action_dim = 3 if not transport else 6
    print(f"Horizon: {horizon}, Action Dim: {action_dim}")
    mlp = MLP(input_dim=action_dim, hidden_dim=hidden_dim, output_dim=action_dim).to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        mlp.train()
        total_loss = 0
        means = 0
        with tqdm.tqdm(dataloader, desc=f"Train Epoch {epoch+1}", disable=noprint, leave=False) as tepoch:
            for batch in tepoch:
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                gt_actions = batch['action']
                nactions = normalizer['action'].normalize(gt_actions)
                means += torch.abs(nactions).mean()
                # nactions = get_sampled(nactions) # NOISE or CORR
                
                if not transport:
                    inputs = nactions[:, horizon-2:horizon-1, :3].reshape(nactions.shape[0], -1).to(device)  # Flatten all but last step
                else:
                    inputs = nactions[:, horizon-2:horizon-1, :3, 10:13].reshape(nactions.shape[0], -1).to(device)  # Flatten all but last step
                
                targets = nactions[:, horizon-1].to(device)  # Last step
                assert len(inputs[0]) == action_dim * NP
                
                preds = mlp(inputs)
                loss = loss_fn(preds, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1}: Training Loss = {total_loss / len(dataloader)}")
        print(means/len(dataloader))

        # Validation loop
        mlp.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}", disable=noprint):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                gt_actions = batch['action']
                nactions = normalizer['action'].normalize(gt_actions)
                # nactions = get_sampled(nactions) # if you want TOY NOISE or TOY CORR

                if not transport:
                    inputs = nactions[:, horizon-2:horizon-1, :3].reshape(nactions.shape[0], -1).to(device)  # Flatten all but last step
                else:
                    inputs = nactions[:, horizon-2:horizon-1, :3, 10:13].reshape(nactions.shape[0], -1).to(device)  # Flatten all but last step
                targets = nactions[:, horizon-1].to(device)
                assert len(inputs[0]) == action_dim * NP
                
                preds = mlp(inputs)
                loss = loss_fn(preds, targets)
                total_val_loss += loss.item()

        # Step the scheduler
        scheduler.step(total_val_loss / len(val_dataloader))

        print(f"Epoch {epoch+1}: Validation Loss = {total_val_loss / len(val_dataloader)}")

if __name__ == '__main__':
    main()

