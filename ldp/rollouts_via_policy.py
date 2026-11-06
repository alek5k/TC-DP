import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import dill
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm  # For progress bar

def print_loss_quantiles(losses, name, quantiles=[0.001,0.005,0.01, 0.05]):
    loss_array = np.array(losses)
    q_values = np.quantile(loss_array, quantiles)
    print(f"Loss Quantiles: {name}")
    for q, val in zip(quantiles, q_values):
        print(f"  {int(q*1000)/10.0}th percentile: {val:.6f}")

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

class ActionDataset(torch.utils.data.Dataset):
    def __init__(self, data, split='train', sequence_length=32, window=None, normalizer=None, mean=None, std=None):
        """
        Args:
            data (numpy array): The normalized data of shape (batch, timesteps, action_dim).
            split (str): 'train' or 'test', to specify the split of the dataset.
            sequence_length (int): The length of the sequence to return.
            mean (float): The mean of the dataset (used for normalization).
            std (float): The standard deviation of the dataset (used for normalization).
        """
        self.data = data
        self.sequence_length = sequence_length
        self.mean = mean
        self.std = std

        # Normalize data if mean and std are provided
        if self.mean is not None and self.std is not None:
            self.data = (self.data - self.mean) / self.std

        if normalizer is not None:
            self.data = normalizer['action'].normalize(self.data)

        # Split data into train and test
        num_batches = self.data.shape[0]
        test_values = len(self.data) // 10
        self.window = self.data.shape[1] - self.sequence_length

        if split == 'train':
            self.batches = self.data[test_values:]
            print(f"Loaded {len(self.batches)} trajectories for training")
        elif split == 'test':
            self.batches = self.data[:test_values]
            print(f"Loaded {len(self.batches)} trajectories for test")
        else:
            raise ValueError("Split must be 'train' or 'test'")

    def __len__(self):
        # Length is number of batches
        return len(self.batches) * self.window

    def __getitem__(self, idx):
        batch = self.batches[idx // self.window]  # Select batch index
        sequence_idx = idx % self.window  # Randomly select starting sequence index between 0 and 100

        # Select the sequence from the batch
        sequence = batch[sequence_idx:sequence_idx+self.sequence_length, :]

        # Return the sequence (normalized)
        # print(sequence.shape)
        return torch.tensor(sequence, dtype=torch.float32)

def compute_normalization_values(data):
    """
    Compute the mean and standard deviation for normalization.
    Args:
        data (numpy array): The data to compute statistics over (batch, timesteps, action_dim).
    Returns:
        mean (float): The mean of the dataset.
        std (float): The standard deviation of the dataset.
    """
    # Flatten the data to compute mean and std over the entire dataset
    flattened_data = data.reshape(-1, data.shape[-1])
    mean = np.mean(flattened_data, axis=0)
    std = np.std(flattened_data, axis=0)
    return mean, std

def create_datasets(path, checkpoint):
    """
    Load the merged actions from the given path, compute normalization values,
    and create train and test datasets.
    Args:
        path (str): Path to the merged_actions.pkl file.
    Returns:
        train_dataset (ActionDataset): The training dataset.
        test_dataset (ActionDataset): The testing dataset.
    """
    # Load the merged actions from the .pkl file
    with open(path, 'rb') as f:
        merged_actions = pickle.load(f)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()

    # Create the train and test datasets
    train_dataset = ActionDataset(merged_actions, split='train', normalizer=normalizer)
    test_dataset = ActionDataset(merged_actions, split='test', normalizer=normalizer)

    return train_dataset, test_dataset

def train(transport, model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=300, use_tqdm=True):
    train_losses = []
    test_losses = []

    # If using tqdm, we wrap the dataloaders in tqdm to show the progress bar
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = []
        epoch_test_loss = []
        magnitudes = 0

        # Training phase
        for batch in tqdm(train_loader, disable=not use_tqdm, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
            if transport:
                inputs = torch.cat([batch[:, 15:16, :3], batch[:, 15:16, 10:13]], dim=-1)
                targets = torch.cat([batch[:, 16, :3], batch[:, 16, 10:13]], dim=-1)
            else:
                inputs = batch[:, 15:16, :3]
                targets = batch[:, 16, :3]

            # breakpoint()
            magnitudes += torch.abs(targets).mean()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs.view(inputs.size(0), -1))  # Flatten input for the MLP
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())
        print(magnitudes / len(train_loader))

        # Testing phase
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, disable=not use_tqdm, desc=f"Epoch {epoch+1}/{num_epochs} - Test"):
                if transport:
                    inputs = torch.cat([batch[:, 15:16, :3], batch[:, 15:16, 10:13]], dim=-1)
                    targets = torch.cat([batch[:, 16, :3], batch[:, 16, 10:13]], dim=-1)
                else:
                    inputs = batch[:, 15:16, :3]
                    targets = batch[:, 16, :3]

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs.view(inputs.size(0), -1))
                loss = criterion(outputs, targets)

                epoch_test_loss.append(loss.item())

        # Average the losses
        # epoch_train_loss /= len(train_loader)
        # epoch_test_loss /= len(test_loader)

        # train_losses.append(epoch_train_loss)
        # test_losses.append(epoch_test_loss)
        mean_train_loss = sum(epoch_train_loss) / len(train_loader)
        mean_test_loss = sum(epoch_test_loss) / len(test_loader)

        # Step the scheduler
        scheduler.step(mean_test_loss)

        # Print the losses at the end of each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {mean_train_loss:.7f}, Test Loss: {mean_test_loss:.7f}")

        print("-----------")

    return train_losses, test_losses

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Train MLP on merged actions')
    parser.add_argument('path', type=str, help='Path to the merged_actions.pkl file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint to run')
    parser.add_argument('--disable_tqdm', action='store_true', help='Disable the TQDM progress bar')
    parser.add_argument('--transport', action='store_true', help='Use 2-arm setup instead of 1-arm')
    args = parser.parse_args()

    # Create datasets
    train_dataset, test_dataset = create_datasets(args.path, args.checkpoint)

    # DataLoader with batch_size 256
    print(len(train_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Model parameters
    input_dim = 3 if not args.transport else 6  # First 16 time steps, 32 actions each (flattened for MLP)
    hidden_dim = 64
    output_dim = 3 if not args.transport else 6  # 32 actions at time step 16
    model = MLP(input_dim, hidden_dim, output_dim)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Train the model
    train_losses, test_losses = train(args.transport, model, train_loader, test_loader, criterion, 
                                      optimizer, scheduler, device, num_epochs=20, use_tqdm=not args.disable_tqdm)

if __name__ == "__main__":
    main()