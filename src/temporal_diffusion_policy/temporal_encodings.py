import numpy as np
from typing import Optional, Tuple, Dict, Any

def calculate_growth_rate_parameter(max_steps: int, target_percent: float = 0.99) -> float:
    '''
    max_steps: how many timesteps of a value should reach saturation
    target_percent: what percent of saturation to reach at max_steps
    Returns alpha/omega parameter for saturating accumulator
    0 < target_percent < 1

    This is in the paper appendix.
    '''
    # calculate alpha/omega based on max value, where 0.99 corresponds to the max value
    alpha = float(1 - (1 - target_percent) ** (1 / max_steps)) # if alpha small, s rises slowly, early steps well resolved. Rises by approximately alpha at the start.
    return alpha

def progress_positional_encoding_transformer(t: int | np.ndarray, d_model: int, power_value: float = 10000.0, T: int = None):
    """
    Attention-style sinusoidal progress positional encoding.
    Since timestep is positive integers, we don't need to store history - we can compute directly from t.

    Args:
        t: current timestep - int or array-like shape (...,)
        d_model: embedding dimension (must be even)
        power_value: base used in the denominator (default: 10000 corresponds to original transformer paper)
        (Optional) T: normalisation constant, most commonly episode length (> 1). If provided, will normalize timestep by this number to get progress in [0, 1] prior to encoding. Not used in paper.

    Returns:
        pe: array of shape (..., d_model)
    """
    assert d_model % 2 == 0, "d_model must be even"
    assert T is None or T > 1, "Episode length T must be greater than 1."
    half = d_model // 2
    t = np.asarray(t, dtype=np.float32)
    i = np.arange(half, dtype=np.float32)
    denom = np.power(power_value, (2.0 * i) / d_model).astype(np.float32) 
    if T is not None:
        t = t / (T - 1.0)  # normalize to [0, 1]
    angles = t[..., None] / denom[None, :]         # (..., half)
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def progress_positional_encoding_manual(t: int | np.ndarray, freqs=(1, 2, 4, 8), T: int = None):
    """
    Manual multi-frequency sin/cos progress positional encoding.
    Not presented in paper and not used.

    Args:
        t: current timestep (int) or array-like of shape (...,)
        freqs: iterable of frequencies
        T: normalisation constant (most commonly episode length) (> 1). If provided, will normalize timestep by this number to get progress in [0, 1] prior to encoding.
    Returns:
        pe: array of shape (..., 2*len(freqs))
    """
    freqs = np.asarray(freqs, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    assert T is None or T > 1, "Episode length T must be greater than 1."

    if T is not None:
        t = t / (T - 1.0)                               # (...,) in [0,1]
    phase = 2.0 * np.pi * t[..., None] * freqs     # (..., F)
    return np.concatenate([np.sin(phase), np.cos(phase)],  axis=-1)


def progress_saturalising_encoding(t: int | np.ndarray, omega: float = 0.05):
    """
    Saturating progress encoding that asymptotically approaches 1.0.
    Since timestep is positive integers, we don't need to store history - we can compute directly from t.

    Args:
        t: current timestep (int) or array-like of shape (...,)
        omega: growth rate parameter (smaller omega means slower growth, more resolution at early steps). This is `omega` in the paper.
    Returns:
        pe: array of shape (..., 1)
    """
    t = np.asarray(t, dtype=np.float32)
    progress = np.zeros_like(t, dtype=np.float32)

    for idx in np.ndindex(t.shape):
        for step in range(int(t[idx]) + 1):
            progress[idx] = progress[idx] + (1.0 - progress[idx]) * omega if step > 0 else omega

    return progress # progress[..., None]  # (..., 1)


class IdlenessEncoder:
    def __init__(self, vmax: float, rest_thresh: float = 0.03, alpha: float = 0.05):
        """
        This is the Idleness Encoding.

        Args:
            vmax: maximum expected velocity for normalization
            rest_thresh: threshold (in normalized velocity units) below which the agent is considered idle. Theta in paper.
            alpha: growth rate parameter for the saturating accumulator
        """
        self.vmax = vmax
        self.rest_thresh = rest_thresh
        self.alpha = alpha
        self.rest = 0.0

    def reset(self):
        self.rest = 0.0

    def step(self, v: float) -> float:
        """
        Update with scalar velocity v (can be signed). Returns rest in [0,1].

        rest[n] = rest[n-1] + (1 - rest[n-1]) * alpha

        rearranging:
        rest[n] = rest[n-1] + alpha - (alpha * rest[n-1])
        rest[n] = alpha + (1 - alpha) * rest[n-1]

        where x is 1
        x[n]=1 is a step input
        Since we are computing the step response of a first-order system

        3 properties:
            Exponential saturation toward 1
            Hard reset on condition violation
            Memory with a time constant
        """
        s = float(abs(v))
        v_norm = s / self.vmax
        is_resting = v_norm < self.rest_thresh # indicator function in paper

        if is_resting:
            self.rest = self.rest + (1.0 - self.rest) * self.alpha
        else:
            self.rest = 0.0

        return self.rest

    def run_episode(self, v_series) -> np.ndarray:
        """Vector wrapper: runs step() across a 1D sequence."""
        self.reset()
        out = np.zeros(len(v_series), dtype=np.float32)
        for i, v in enumerate(v_series):
            out[i] = self.step(v)
        return out
    

class PauseInducedPerturbation:
    """
    Implementation of
    https://arxiv.org/pdf/2508.15669

    One-stop implementation of:
      (1) Idling detection: consecutive timesteps where ||q_t - q_{t-1}||_2 < epsilon for > window_size steps.
      (2) Test-time perturbation: if idle, override action with a_delta = sigma * q_t + (1 - sigma) * q_0 (position-control joint target interpolation toward initial joints).

      q_min_max: optional joint limits to normalize the joint values for idle detection. This does not exist in the original paper, 
      but in order to re-use this class for robot position control environments, we add this option.

      It's not clear in the original paper if the perturbation is done just once when idle is detected, or if it's done continuously while idling.
      The parameter "reset_on_perturbation" can be modified to not reset the idle detector after perturbation is applied.
    """

    def __init__(self, epsilon: float = 0.06, window_size: int = 8, sigma: float = 0.6, q_min: Optional[np.ndarray] = None, q_max: Optional[np.ndarray] = None, reset_on_perturbation: bool = True):
        self.epsilon = float(epsilon)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.qmin = np.asarray(q_min) if q_min is not None else None
        self.qmax = np.asarray(q_max) if q_max is not None else None
        assert 0.0 <= self.sigma <= 1.0
        self.total_steps: int = 0 # just for tracking
        self.reset_on_perturbation = reset_on_perturbation

    def reset(self, q0: np.ndarray):
        self.consecutive_below_thresh: int = 0
        self.q_prev: Optional[np.ndarray] = None
        self.q0 = np.asarray(q0).copy()
        self.total_steps = 0

    def _update_idle(self, q_t: np.ndarray) -> Tuple[bool, float]:
        """
        Updates idle detector state from current joint positions q_t.
        Returns (idle_now, l2_change).
        """
        q_t = np.asarray(q_t)

        if self.q_prev is None:
            self.q_prev = q_t.copy()

        dq = q_t - self.q_prev
        if self.qmin is not None and self.qmax is not None:
            dq = dq / (self.qmax - self.qmin + 1e-12)
        l2_change = np.linalg.norm(dq)

        self.q_prev = q_t.copy()

        if l2_change < self.epsilon:
            self.consecutive_below_thresh += 1
        else:
            self.consecutive_below_thresh = 0

        idle_now = self.consecutive_below_thresh > self.window_size

        return idle_now, l2_change

    def _perturb(self, q_t: np.ndarray) -> np.ndarray:
        """Compute a_delta = sigma*q_t + (1-sigma)*q0."""
        if self.q0 is None:
            raise RuntimeError("Call start_episode(q0) before calling step().")
        q_t = np.asarray(q_t)
        return self.sigma * q_t + (1.0 - self.sigma) * self.q0

    def step(self, q_t: np.ndarray, a_policy: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main API:
          - q_t: current joint positions (D,)
          - a_policy: policy-predicted joint target positions (D,)
        Returns:
          - action_to_execute (D,)
          - info dict with diagnostics
        """
        q_t = np.asarray(q_t)
        a_policy = np.asarray(a_policy)

        idle_now, l2_change = self._update_idle(q_t)
        # print("l2 magnitude", l2_change)
        should_perturb = idle_now 

        if should_perturb:
            a_exec = self._perturb(q_t)
            if self.reset_on_perturbation:
                self.consecutive_below_thresh = 0
        else:
            a_exec = a_policy

        info = {
            "idle_now": idle_now,
            "should_perturb": should_perturb,
            "l2_change": l2_change,
            "consecutive_below_thresh": self.consecutive_below_thresh,
        }
        self.total_steps += 1
        return a_exec, info



def binary_idle_indicator(
    velocity: np.ndarray, vmax: float, threshold: float = 0.03
) -> np.ndarray:
    """Return instantaneous ``1[|v| / vmax < threshold]`` as ``(N, 1)``.

    For vector velocities, ``|v|`` is the L2 magnitude.  This function never
    accumulates, smooths, or otherwise uses prior timesteps.
    """
    magnitude = velocity_magnitude(velocity)
    if vmax <= 0:
        raise ValueError("vmax must be positive for the binary idle indicator")
    return (magnitude / vmax < threshold).astype(np.float32)[:, None]

def velocity_vmax(velocity: np.ndarray) -> float:
    """Return the maximum observed velocity magnitude."""
    return float(np.max(velocity_magnitude(velocity)))
