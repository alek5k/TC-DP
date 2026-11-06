
## Temporal Encodings

This project implements several temporal encoding techniques to help policies reason about time and detect idle states.

### 1. Idleness Encoding (ID)

**Purpose**: Detects when the agent is at rest (low velocity) and accumulates a "idleness" signal.

**Method**: 
- Monitors agent velocity magnitude ||v|| at each timestep
- Compares velocity to rest threshold θ (default: 0.03 in normalized units)
- Accumulates idleness when velocity is below threshold:
  ```
  rest[t] = rest[t-1] + (1 - rest[t-1]) × α    if ||v|| < θ
  rest[t] = 0                                   otherwise (hard reset)
  ```
- Produces exponential saturation toward 1.0 during idle periods
- Output: Single scalar in [0, 1] representing accumulated idle time
- The encodings are fed to the policy network as conditioning inputs.

**Usage**:
```bash
python src/temporal_diffusion_policy/training.py --use_idleness_encoding True --idleness_encoding_alpha 0.05
```

**Key Parameters**:
- `idleness_encoding_alpha` (α): Growth rate (smaller = slower saturation, more early-step resolution)
- Velocity threshold (θ): 0.03 in normalized velocity units (hardcoded in dataset)

### 2. Saturating Progress Encoding (SaP)

**Purpose**: Provides episode progress information with non-linear saturation toward 1.0.

**Method**:
- Converts discrete timestep t into continuous progress signal
- Uses saturating accumulator: `progress[t] = progress[t-1] + (1 - progress[t-1]) × ω`
- Asymptotically approaches 1.0, providing high resolution at early timesteps
- Output: Single scalar in [0, 1]

**Usage**:
```bash
python src/temporal_diffusion_policy/training.py --use_saturating_progress_encoding True --saturating_progress_encoding_omega 0.05
```

**Key Parameters**:
- `saturating_progress_encoding_omega` (ω): Growth rate parameter
- If `None`, automatically computed from dataset max episode length

Larger ω results in faster saturation 

### 3. Sinusoidal Progress Encoding (SiP)

**Purpose**: Transformer-style positional encoding adapted for temporal progress.

**Method**:
- Multi-frequency sin/cos encoding of timestep position
- Similar to original Transformer positional encoding (Vaswani et al., 2017)
- For dimension d and timestep t:
  ```
  PE(t, 2i)   = sin(t / 10000^(2i/d))
  PE(t, 2i+1) = cos(t / 10000^(2i/d))
  ```
- Output: Vector of size `d_model` (e.g., 4, 8, 32, 64, 128, 256)

**Usage**:
```bash
python src/temporal_diffusion_policy/training.py \
    --use_sinusoidal_progress_encoding True \
    --sinusoidal_progress_encoding_dim 32
```

**Key Parameters**:
- `sinusoidal_progress_encoding_dim`: Embedding dimension (must be even)
- Higher dimensions encode more frequency information

**Trade-off**: Larger dimensions = richer representation but more conditioning parameters (does not seem to hurt, actually generally helps performance with larger dimensions up to a point)

### 4. Pause-Induced Perturbation (PIP)

**Purpose**: Test-time intervention to break out of idle states (inference only).

**Method** (from [Chen et al., 2025](https://arxiv.org/pdf/2508.15669)):
1. **Idle Detection**: Tracks consecutive timesteps where joint position change `||q[t] - q[t-1]|| < ε`
2. **Perturbation**: If idle for > `window_size` steps, overrides policy action with:
   ```
   a_perturbed = σ × q[t] + (1-σ) × q[0]
   ```
   where q[0] is initial joint configuration

**Usage**:
```bash
python src/temporal_diffusion_policy/inference.py \
    --checkpoint_path <path> \
    --use_pause_induced_perturbation True \
    --pip_epsilon 0.03 \
    --pip_window_size 20 \
    --pip_sigma 0.99
```

**Key Parameters**:
- `pip_epsilon` (ε): Position change threshold for idle detection
- `pip_window_size`: Consecutive idle steps before perturbation
- `pip_sigma` (σ): Interpolation weight (1.0 = current position, 0.0 = initial position)

**Note**: PIP is a baseline method for comparison. It does not require training with temporal encodings.
