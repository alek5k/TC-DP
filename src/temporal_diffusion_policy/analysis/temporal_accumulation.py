from dataclasses import dataclass, asdict
# Analysis package module.
import os
from pathlib import Path
from typing import Union
import draccus
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from temporal_diffusion_policy.dataset_generation import NormalizationType, TemporalConditioningDataset, iter_episodes

@dataclass
class AnalysisConfig:
    pred_horizon: int = 16
    obs_horizon: int = 2
    action_horizon: int = 8
    dataset_path: str = "data/demonstration/wait_at_goal_dataset_v4.zarr"
    
    
    use_idleness_encoding: bool = True
    use_sinusoidal_progress_encoding: bool = True
    sinusoidal_progress_encoding_dim: int = 4
    use_saturating_progress_encoding: bool = True

    plot_max: bool = False
    plot_first_n: int = 3
    plot_specific_episode: Union[None, int] = None

    saturating_progress_encoding_omega: Union[None, float] = 0.047
    idleness_encoding_alpha: Union[None, float] = 0.047
    idleness_encoding_rest_thresh: float = 0.03


font_size = 24
tick_label_size = 14
figheight = 4.8
figwidth = 6.4
OUTPUT_DIR = Path("outputs/analysis/accumulation")

# sudo apt-get update
# sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super

matplotlib.rcParams.update({
    "font.size": font_size,          # base text
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": tick_label_size,
    "ytick.labelsize": tick_label_size,
    "legend.fontsize": tick_label_size,
    
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{times}",
    "font.family": "serif",   # LaTeX will use Computer Modern by default
    "axes.unicode_minus": False,

})


def plot_idleness_encoding(data, label="idleness"):
    '''
    Expects data shape (N,) where N is value
    0 to 1 time at rest from velocity
    '''
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax.plot(data, linewidth=2.5, color="#1f77b4")
    ax.axhline(1.0, linestyle=":", linewidth=1.2, alpha=1.0)
    # ax.set_title('Saturating Time at Rest from Velocity Encoding')
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Idleness Encoding $r_t$')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(pad=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"idleness_{label}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"idleness_{label}.pdf", dpi=300)

def plot_saturating_progress_encoding(data, label="saturating_progress"):
    '''
    Expects data shape (N,) where N is value
    0 to 1 progress from step count
    '''
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax.plot(data, linewidth=2.5, color="#1f77b4")
    ax.axhline(1.0, linestyle=":", linewidth=1.2, alpha=1.0)
    # ax.set_title('Saturating Progress from Step Count Encoding')
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Progress Encoding $p_t$')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout(pad=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"saturating_progress_{label}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"saturating_progress_{label}.pdf", dpi=300)

def plot_sinusoidal_progress_encoding(data, label="sinusoidal_progress"):
    '''
    Expects data shape (N, D) where N is value, D is dimension of encoding
    0 to D/2 - sin components
    D/2 to D - cos components
    '''
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    num_dims = data.shape[1]
    half = num_dims // 2
    for dim in range(num_dims):
        sin_or_cos = 'sin' if dim < half else 'cos'
        i = dim % half
        ax.plot(data[:, dim], label=f'$i={i}$ ({sin_or_cos})', linewidth=2.5)
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Progress Encoding $g_t$')
    ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylim(-1.05, 1.05)
    if num_dims <= 8:
        ax.legend(loc='lower right')
        # fig.tight_layout()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"progress_{label}_dim{num_dims}.png", dpi=300)
    fig.savefig(OUTPUT_DIR / f"progress_{label}_dim{num_dims}.pdf", dpi=300)

@draccus.wrap()
def main(cfg: AnalysisConfig):
    
    if not os.path.exists(cfg.dataset_path):
        raise FileNotFoundError(f"Dataset path {cfg.dataset_path} does not exist.")

    dataset = TemporalConditioningDataset(
        dataset_path=cfg.dataset_path,
        pred_horizon=cfg.pred_horizon,
        obs_horizon=cfg.obs_horizon,
        action_horizon=cfg.action_horizon,
        use_idleness_encoding=cfg.use_idleness_encoding,
        use_sinusoidal_progress_encoding=cfg.use_sinusoidal_progress_encoding,
        sinusoidal_progress_encoding_parameters={'d_model': cfg.sinusoidal_progress_encoding_dim, 'power_value': 10000.0, 'normalize_by_episode_length': False},
        use_saturating_progress_encoding=cfg.use_saturating_progress_encoding,
        idleness_encoding_alpha=cfg.idleness_encoding_alpha,
        saturating_progress_encoding_omega=cfg.saturating_progress_encoding_omega,
        idleness_encoding_rest_thresh=cfg.idleness_encoding_rest_thresh

    )
    # save training data statistics (min, max) for each dim
    # stats = dataset.stats
    train_data = dataset.full_train_data

    if cfg.use_idleness_encoding:
        if cfg.plot_max:
            # train_data['idleness'].shape (14008, 1)
            ep_with_max_idleness = -1
            ep_with_max_idleness_indices = (-1, -1)
            curr_max = -1
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                episode_idleness = train_data['idleness'][start_idx:end_idx]
                max_idleness = np.max(episode_idleness)
                if max_idleness > curr_max:
                    ep_with_max_idleness = i
                    ep_with_max_idleness_indices = (start_idx, end_idx)
                    curr_max = max_idleness
            
            max_data = train_data['idleness'][ep_with_max_idleness_indices[0]:ep_with_max_idleness_indices[1]]
            plot_idleness_encoding(max_data, label=f"episode{ep_with_max_idleness}")
        
        if cfg.plot_first_n:
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                episode_data = train_data['idleness'][start_idx:end_idx]
                plot_idleness_encoding(episode_data, label=f"episode{i}")
                if i >= cfg.plot_first_n - 1:
                    break
        
        if cfg.plot_specific_episode is not None:
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                if i == cfg.plot_specific_episode:
                    episode_data = train_data['idleness'][start_idx:end_idx]
                    plot_idleness_encoding(episode_data, label=f"specific_episode{i}")

    if cfg.use_saturating_progress_encoding:
        
        if cfg.plot_max:
            ep_with_max_progress = -1
            ep_with_max_progress_indices = (-1, -1)
            curr_max = -1
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                episode_progress = train_data['saturating_progress_encoding'][start_idx:end_idx]
                max_progress = np.max(episode_progress)
                if max_progress > curr_max:
                    ep_with_max_progress = i
                    ep_with_max_progress_indices = (start_idx, end_idx)
                    curr_max = max_progress
            
            max_data = train_data['saturating_progress_encoding'][ep_with_max_progress_indices[0]:ep_with_max_progress_indices[1]]
            plot_saturating_progress_encoding(max_data, label=f"episode{ep_with_max_progress}")
        
        if cfg.plot_first_n:
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                episode_data = train_data['saturating_progress_encoding'][start_idx:end_idx]
                plot_saturating_progress_encoding(episode_data, label=f"episode{i}")
                if i >= cfg.plot_first_n - 1:
                    break
        
        if cfg.plot_specific_episode is not None:
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                if i == cfg.plot_specific_episode:
                    episode_data = train_data['saturating_progress_encoding'][start_idx:end_idx]
                    plot_saturating_progress_encoding(episode_data, label=f"specific_episode{i}")
                
    if cfg.use_sinusoidal_progress_encoding:

        if cfg.plot_first_n:
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                episode_data = train_data['sinusoidal_progress_encoding'][start_idx:end_idx]
                plot_sinusoidal_progress_encoding(episode_data, label=f"episode{i}")
                if i >= cfg.plot_first_n - 1:
                    break
    
    if cfg.plot_specific_episode is not None:
            for i, (start_idx, end_idx) in enumerate(iter_episodes(dataset.episode_ends)):
                if i == cfg.plot_specific_episode:
                    episode_data = train_data['sinusoidal_progress_encoding'][start_idx:end_idx]
                    plot_sinusoidal_progress_encoding(episode_data, label=f"specific_episode{i}")
        
if __name__ == '__main__':
    main()
