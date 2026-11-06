from dataclasses import dataclass, asdict
# Analysis package module.
import os
from pathlib import Path
from typing import Union
import draccus
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # or use cv2.imread
import bisect

from temporal_diffusion_policy.dataset_generation import NormalizationType, TemporalConditioningDataset, iter_episodes
from temporal_diffusion_policy.replay_buffer import ReplayBuffer

@dataclass
class AnalysisConfig:
    demo_dataset_path: str = "data_saved/demonstration/lift_qa_v2.zarr"
    dataset_path: str = "data_saved/inference_compressed/LiftQA_b08d56_ckpt20260122_000453_.zarr"
    use_highres_images: bool = True

font_size = 16
tick_label_size = 14
annotation_size = 16
figheight = 7.4
figwidth = 6.4
space_between_subplots = 0.08
subplot_height_ratios = [3.0, 2.2]
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

COLOR_DEMO = "#1f77b4"
COLOR_INF  = "#ff7f0e"

from matplotlib.patches import ConnectionPatch

def connect(ax_top, ax_img, x_top, y_top=0.0, x_img_frac=0.5, color="red", lw=2.0, ls=":", zorder=-10):
    """
    Draw a connector from ax_top at (x_top, y_top) in data coords
    to ax_img at (x_img_frac, 1.0) in axes fraction coords (top edge of image).
    """
    con = ConnectionPatch(
        xyA=(x_top, y_top), coordsA=ax_top.transData,          # in top plot data coords
        xyB=(x_img_frac, 1.0), coordsB=ax_img.transAxes,      # top of image (axes coords)
        arrowstyle="-",                                        # line only
        linestyle=ls,
        lw=lw,
        color=color,
        zorder=zorder,
        clip_on=False,
    )
    ax_top.add_artist(con)

def draw_line(ax, x0, y0, x1, y1, color="red", lw=2.0, ls=":", zorder=-10):
    con = ConnectionPatch(
        xyA=(x0, y0), coordsA=ax.transData,          # in top plot data coords
        xyB=(x1, y1), coordsB=ax.transData,      # top of image (axes coords)
        arrowstyle="-",                                        # line only
        linestyle=ls,
        lw=lw,
        color=color,
        zorder=zorder,
        clip_on=False,
    )
    ax.add_artist(con)


def plot_velocity(data_demo, data_inf, label="velocity"):
    '''
    Expects data shape (N,) where N is value
    0 to 1 progress from step count
    '''
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
    ax.plot(data_demo, linewidth=2.5, color=COLOR_DEMO, alpha=0.7)
    ax.plot(data_inf, linewidth=2.5, color=COLOR_INF, alpha=0.7)
    # ax.axhline(1.0, linestyle=":", linewidth=1.2, alpha=1.0)
    # ax.set_title('Saturating Progress from Step Count Encoding')
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Agent Velocity $v_t$')
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_ylim(-0.05, 1.05)
    ax.legend(['Demonstration', 'Diffusion Policy'], loc='upper right')
    fig.tight_layout(pad=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    
    return fig, ax
    # fig.savefig(f"analysis_accumulation/velocity_{label}.png", dpi=300)
    # fig.savefig(f"analysis_accumulation/saturating_progress_{label}.pdf", dpi=300)


def mean_over_timestep(episodes: list[np.ndarray]):
    # episodes: list of 1D arrays, variable length
    max_T = max(len(ep) for ep in episodes)
    V = np.full((len(episodes), max_T), np.nan, dtype=float)

    for i, ep in enumerate(episodes):
        ep = np.asarray(ep, dtype=float).reshape(-1)  # ensure 1D
        V[i, :len(ep)] = ep

    mean = np.nanmean(V, axis=0)
    std  = np.nanstd(V, axis=0)
    n    = np.sum(~np.isnan(V), axis=0)  # how many eps contributed per t
    sem  = std / np.sqrt(np.maximum(n, 1))
    return mean, sem, n


def mean_over_progress(episodes: list[np.ndarray], K: int = 200, *,  as_magnitude: bool = False):
    """
    Compute mean velocity over *normalised episode progress*.

    Episodes can be variable length.
    Returns:
        mean: (K,) mean velocity vs progress
        sem:  (K,) standard error of the mean vs progress
        n:    (K,) number of valid episodes contributing per progress bin
        grid: (K,) progress grid in [0,1]
    """
    grid = np.linspace(0.0, 1.0, K, dtype=float)
    V = np.full((len(episodes), K), np.nan, dtype=float)

    for i, ep in enumerate(episodes):
        ep = np.asarray(ep, dtype=float)

        # If velocity is vector-valued per timestep (T, D), optionally convert to magnitude
        if ep.ndim == 2:
            if as_magnitude:
                ep = np.linalg.norm(ep, axis=-1)
            else:
                raise ValueError(
                    f"Episode {i} has shape {ep.shape}. "
                    "Set as_magnitude=True to reduce (T,D)->(T,)."
                )

        ep = ep.reshape(-1)  # ensure (T,)
        T = len(ep)

        if T < 2:
            continue  # can't interpolate

        x = np.linspace(0.0, 1.0, T, dtype=float)
        V[i] = np.interp(grid, x, ep)

    mean = np.nanmean(V, axis=0)
    std  = np.nanstd(V, axis=0)
    n    = np.sum(~np.isnan(V), axis=0)
    sem  = std / np.sqrt(np.maximum(n, 1))

    return mean, sem, n, grid



def top5_demo_matches_by_inf_length(velocities_demo, velocities_inf):
    K = 5  # hardcoded top-k

    # (demo_len, demo_ep) sorted by length
    demo_lens = sorted((len(ep), i) for i, ep in enumerate(velocities_demo))
    demo_T = [T for T, _ in demo_lens]

    matches = {}  # inf_ep -> dict

    for inf_ep, inf in enumerate(velocities_inf):
        inf_len = len(inf)
        k = bisect.bisect_left(demo_T, inf_len)

        left, right = k - 1, k
        top5 = []

        while len(top5) < K and (left >= 0 or right < len(demo_lens)):
            candidates = []

            if left >= 0:
                demo_len, demo_ep = demo_lens[left]
                candidates.append(("L", abs(demo_len - inf_len), demo_ep, demo_len))

            if right < len(demo_lens):
                demo_len, demo_ep = demo_lens[right]
                candidates.append(("R", abs(demo_len - inf_len), demo_ep, demo_len))

            side, diff, demo_ep, demo_len = min(candidates, key=lambda x: x[1])

            top5.append({
                "demo_ep": demo_ep,
                "demo_len": demo_len,
                "diff": diff,
            })

            if side == "L":
                left -= 1
            else:
                right += 1

        matches[inf_ep] = {
            "inf_ep": inf_ep,
            "inf_len": inf_len,
            "best": top5[0] if top5 else None,
            "top5": top5,
        }

    return matches

@draccus.wrap()
def main(cfg: AnalysisConfig):
    
    if not os.path.exists(cfg.dataset_path) or not os.path.exists(cfg.demo_dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_buffer = ReplayBuffer.create_from_path(cfg.demo_dataset_path, 'r')
    inf_buffer = ReplayBuffer.create_from_path(cfg.dataset_path, 'r')

    velocities_demo = demo_buffer.get_episode(20)['agent_velocity']
    velocities_inf = inf_buffer.get_episode(88)['agent_velocity']

    # save images for the bottom strip
    if cfg.use_highres_images:
        img_paths = [
            str(OUTPUT_DIR / "highres_A.png"),
            str(OUTPUT_DIR / "highres_B.png"),
            str(OUTPUT_DIR / "highres_C.png"),
        ]
    else:
        frames_to_save = [25, 105, 305]
        img_paths = []
        for f in frames_to_save:
            img = inf_buffer.get_episode(88)['full_image'][f]
            # (C, H, W) to (H, W, C) if needed
            img = np.transpose(img, (1, 2, 0))
            frame_path = OUTPUT_DIR / f"frame_{f}.png"
            plt.imsave(frame_path, img)
            img_paths.append(str(frame_path))
        
    # img_paths = [
    #     "analysis_accumulation/frame_0.png",
    #     "analysis_accumulation/frame_105.png",
    #     "analysis_accumulation/frame_305.png",
    # ]
    imgs = [mpimg.imread(p) for p in img_paths]   # list of HxWxC arrays

    fig = plt.figure(figsize=(figwidth, figheight))
    
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=subplot_height_ratios, hspace=space_between_subplots)

    ax = fig.add_subplot(gs[0])

    # bottom: a sub-grid with 1 row x 3 cols
    gs_imgs = gs[1].subgridspec(1, 3, wspace=0.05)
    ax_img0 = fig.add_subplot(gs_imgs[0, 0])
    ax_img1 = fig.add_subplot(gs_imgs[0, 1])
    ax_img2 = fig.add_subplot(gs_imgs[0, 2])

    ax.plot(velocities_demo, linewidth=2.5, color=COLOR_DEMO, alpha=0.7)
    ax.plot(velocities_inf, linewidth=2.5, color=COLOR_INF, alpha=0.7)
    # ax.axhline(1.0, linestyle=":", linewidth=1.2, alpha=1.0)
    # ax.set_title('Saturating Progress from Step Count Encoding')
    ax.set_xlabel('Timestep $t$')
    ax.set_ylabel('Agent Velocity $v_t$')
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # ax.set_ylim(-0.05, 1.05)
    ax.legend(['Demonstration', 'Diffusion Policy'], loc='upper right')
    fig.tight_layout(pad=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    
    

    # Interval (in timesteps)
    x0, x1 = 105, 260
    y_bracket = -0.04   # y-position of the bracket line
    y_text    = -0.07   # y-position of the label

    # 1) draw |--------| style bracket
    ax.annotate(
        "",                      # no text, just the bracket
        xy=(x0, y_bracket),
        xytext=(x1, y_bracket),
        xycoords="data",
        textcoords="data",
        arrowprops=dict(
            arrowstyle="|-|",
            lw=2,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=8, # the bars at the end
        ),
    )
    # 2) arrowheads in the middle
    ax.annotate(
        "",
        xy=(x0, y_bracket), xytext=(x1, y_bracket),
        arrowprops=dict(arrowstyle="<|-|>", lw=2, ec="black", shrinkA=0, shrinkB=0, fc=COLOR_INF), # , mutation_scale=12
    )

    # 2) add label centered under the bracket
    ax.text(
        (x0 + x1) / 2,
        y_text,
        "Failed/delayed\nactuation",
        ha="center",
        va="top",
        fontsize=annotation_size,
        # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.5),
    )

    ax.annotate(
        "Trajectory restarting",
        xy=(305, 0.00),         # arrow points to restart onset
        xycoords="data",
        xytext=(305, -0.2),    # label below 0
        textcoords="data",
        fontsize=annotation_size,
        ha="center",
        va="top",
        arrowprops=dict(arrowstyle="-|>", lw=2, fc=COLOR_INF, mutation_scale=20),
        # bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1.5),
    )
    ymin, ymax = ax.get_ylim()
    new_ymin = ymin-0.3
    new_ymax = ymax+0.1
    ax.set_ylim(new_ymin, new_ymax)

    color = "#999999"
    lw = 1.0
    ls = "dotted"
    
    connect(ax, ax_img0, x_top=20, y_top=new_ymin, x_img_frac=0.55, color=color, lw=lw, ls=ls)
    draw_line(ax, 20, new_ymin, 20, 0.0, color=color, lw=lw, ls=ls)

    connect(ax, ax_img1, x_top=105, y_top=new_ymin, x_img_frac=0.15, color=color, lw=lw, ls=ls)
    draw_line(ax, 105, new_ymin, 105, 0.0, color=color, lw=lw, ls=ls)
    
    connect(ax, ax_img2, x_top=305, y_top=new_ymin, x_img_frac=0.55, color=color, lw=lw, ls=ls)
    draw_line(ax, 305, new_ymin, 305, 0.0, color=color, lw=lw, ls=ls)



    # ---- bottom image strip ----
    for a, im, path in [
        (ax_img0, imgs[0], img_paths[0]),
        (ax_img1, imgs[1], img_paths[1]),
        (ax_img2, imgs[2], img_paths[2]),
    ]:
        a.imshow(im)
        # extract the frame number from the filename
        # a.set_title(title, fontsize=10)
        # a.axis("off")
        a.set_frame_on(True)
        a.set_xticks([])
        a.set_yticks([])
        # for sp in a.spines.values():
        #     sp.set_visible(True)
        #     sp.set_linewidth(2.0)
        #     sp.set_color("red")   # debug




    fig.savefig(OUTPUT_DIR / "velocity_ep_closest_88_20.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT_DIR / "velocity_ep_closest_88_20.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)

    # fig.savefig(f"analysis_accumulation/velocity_{label}.png", dpi=300)
    # fig.savefig(f"analysis_accumulation/saturating_progress_{label}.pdf", dpi=300)

    # WaitAtGoal episode 179
    # demo_mean, demo_sem, demo_n = mean_over_timestep(velocities_demo)
    # inf_mean,  inf_sem,  inf_n  = mean_over_timestep(velocities_inf)
        
    # plt.plot(t_demo, demo_mean, label=f"demo mean (n0={demo_n[0]})")
    # plt.fill_between(t_demo, demo_mean - demo_sem, demo_mean + demo_sem, alpha=0.2)
    # plt.plot(t_inf, inf_mean, label=f"inf mean (n0={inf_n[0]})")
    # plt.fill_between(t_inf, inf_mean - inf_sem, inf_mean + inf_sem, alpha=0.2)

    # demo_mean, demo_sem, demo_n, prog = mean_over_progress(velocities_demo, K=200)
    # inf_mean,  inf_sem,  inf_n,  _    = mean_over_progress(velocities_inf,  K=200)

    # plot_velocity(demo_mean, inf_mean, label="ep0")

    print()    


if __name__ == '__main__':
    main()
