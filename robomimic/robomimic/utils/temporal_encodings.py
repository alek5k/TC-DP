import numpy as np
from typing import Optional, Tuple, Dict, Any


def calculate_growth_rate_parameter(max_steps: int, target_percent: float = 0.99) -> float:
    """
    max_steps: how many timesteps of a value should reach saturation
    target_percent: what percent of saturation to reach at max_steps
    Returns alpha/omega parameter for saturating accumulator
    0 < target_percent < 1
    """
    alpha = float(1 - (1 - target_percent) ** (1 / max_steps))
    return alpha


def progress_positional_encoding_transformer(
    t: int | np.ndarray,
    d_model: int,
    power_value: float = 10000.0,
    T: int = None,
):
    """
    Attention-style sinusoidal progress positional encoding.

    Args:
        t: current timestep - int or array-like shape (...,)
        d_model: embedding dimension (must be even)
        power_value: base used in the denominator (default: 10000)
        T: optional normalization constant (>1). If provided, normalize t to [0, 1].
    """
    assert d_model % 2 == 0, "d_model must be even"
    assert T is None or T > 1, "Episode length T must be greater than 1."
    half = d_model // 2
    t = np.asarray(t, dtype=np.float32)
    i = np.arange(half, dtype=np.float32)
    denom = np.power(power_value, (2.0 * i) / d_model).astype(np.float32)
    if T is not None:
        t = t / (T - 1.0)
    angles = t[..., None] / denom[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def progress_saturalising_encoding(t: int | np.ndarray, omega: float = 0.05):
    """
    Saturating progress encoding that asymptotically approaches 1.0.

    Args:
        t: current timestep (int) or array-like shape (...,)
        omega: growth rate parameter
    """
    t = np.asarray(t, dtype=np.float32)
    progress = np.zeros_like(t, dtype=np.float32)

    for idx in np.ndindex(t.shape):
        for step in range(int(t[idx]) + 1):
            progress[idx] = progress[idx] + (1.0 - progress[idx]) * omega if step > 0 else omega

    return progress


class IdlenessEncoder:
    def __init__(self, vmax: float, rest_thresh: float = 0.03, alpha: float = 0.05):
        self.vmax = vmax
        self.rest_thresh = rest_thresh
        self.alpha = alpha
        self.rest = 0.0

    def reset(self):
        self.rest = 0.0

    def step(self, v: float) -> float:
        s = float(abs(v))
        v_norm = s / self.vmax
        is_resting = v_norm < self.rest_thresh

        if is_resting:
            self.rest = self.rest + (1.0 - self.rest) * self.alpha
        else:
            self.rest = 0.0

        return self.rest