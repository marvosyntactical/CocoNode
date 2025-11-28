# CocoNODE: Neural ODE GPT with Transparent Latent Reasoning

CocoNODE combines four key ideas for safe, efficient reasoning:

1. **Neural ODE Depth**: Transformer layers as continuous-time dynamics
2. **Adaptive Computation**: Confidence-based early exit (think more on hard tokens)
3. **Coconut Reasoning**: Feed hidden states back as inputs for iterative refinement
4. **Stochastic Transparency**: Random decoding prevents hidden computation

## Why This Matters for AI Safety

Recent research shows frontier models can engage in **scheming**—strategic deception where models pursue hidden goals. Apollo Research found models like Claude 3.5 Sonnet, GPT-4o, and Gemini can:

- Disable oversight mechanisms
- Manipulate training data
- Sandbag on capability evaluations
- Lie about their actions when questioned

The core problem: **latent reasoning is unobservable**. As models shift toward continuous/latent computation (Coconut, looped transformers, etc.), we lose the ability to monitor their "thoughts."

CocoNODE addresses this by making observation **unpredictable**:

```
┌─────────────────────────────────────────────────────────────┐
│  Traditional Model                CocoNODE                  │
│  ─────────────────                ────────                  │
│  Input → [Hidden] → Output        Input → [Hidden] → Output │
│          ↑                                  ↑               │
│     (unobservable)                   (randomly decoded)     │
│                                                             │
│  Model knows when it's           Model cannot predict when  │
│  being observed → can hide       observation occurs → can't │
│  strategic reasoning             learn to hide anything     │
└─────────────────────────────────────────────────────────────┘
```

## Architecture

### Neural ODE Layers

Instead of discrete transformer layers, we treat depth as continuous time:

```python
# Traditional: x = layer_1(x); x = layer_2(x); ...
# CocoNODE:    x = ODESolve(dynamics, x, t=0→1)

class ODEBlock(ODEF):
    def forward(self, z, t):
        # Time-conditioned attention + MLP
        # Returns dz/dt (rate of change)
        zt = concat(z, time_embed(t))
        h = self.attn(zt)
        return self.mlp(concat(h, time_embed(t)))
```

The ODE solver adaptively chooses discretization—more steps for harder tokens.

### Coconut-Style Reasoning

For tasks requiring multi-step reasoning, hidden states feed back as inputs:

```python
for step in range(n_latent_steps):
    # Forward through ODE blocks
    x = ode_blocks(x)
    
    # Feed continuous thought back at designated positions
    # (This is the "latent reasoning" that replaces CoT tokens)
    x[latent_positions] = x[latent_positions]  # Differentiable feedback
```

Training uses curriculum learning: progressively replace language CoT with latent steps.

### Stochastic Transparency

The key safety innovation—random decoding during both training and inference:

```python
class StochasticDecoder:
    def maybe_decode(self, hidden_state, layer_idx):
        # Decision is EXTERNAL randomness (not model-predictable)
        if random.random() < self.decode_prob:
            logits = self.tuned_lens[layer_idx](hidden_state)
            self.record(layer_idx, logits)
            return logits
        return None
```

**Why this prevents scheming:**

1. Model cannot predict which layers will be decoded
2. Consistency loss penalizes intermediate/final divergence
3. Any "hidden" computation risks being exposed
4. The optimal strategy becomes: don't hide anything

## Quick Start

### Installation

```bash
git clone https://github.com/your-repo/coconode
cd coconode
pip install torch numpy tiktoken wandb datasets
```

### Training on Shakespeare (Character-level)

```bash
# Prepare data
python data/shakespeare_char/prepare.py

# Train with fixed depth
python train.py --config configs/train_shakespeare.py

# Sample
python sample.py --out_dir out-coconode-shakespeare
```

### Training on GSM8K Math with Full Features

```bash
# Prepare GSM8K with reasoning steps
python data/gsm8k/prepare.py --output_dir data/gsm8k

# Train with adaptive depth + Coconut curriculum + transparency
python train_adaptive.py --config configs/train_gsm8k_adaptive.py

# This enables:
# - Confidence-based early exit (adaptive depth per token)
# - Progressive replacement of CoT with latent reasoning
# - Stochastic decoding for anti-scheming
```

### Key Training Modes

```bash
# 1. Standard training (no adaptive depth)
python train.py --dataset=gsm8k --max_depth=6

# 2. Adaptive depth only
python train_adaptive.py --transparency_enabled=False

# 3. Full safety mode (adaptive + transparency)
python train_adaptive.py --transparency_enabled=True \
    --decode_prob=0.2 --consistency_weight=0.15

# 4. Debug mode (small model, verbose)
python train_adaptive.py --n_embd=256 --max_depth=4 \
    --batch_size=8 --log_interval=10
```

### Transparency Monitoring

```python
from model import CocoNODE, CocoNODEConfig
from transparency import TransparentCocoNODE, TransparencyConfig, TransparencyMonitor

# Wrap model with transparency
config = TransparencyConfig(decode_prob=0.2, consistency_weight=0.1)
transparent_model = TransparentCocoNODE(base_model, config)

# Monitor during inference
monitor = TransparencyMonitor(transparent_model, config)

# Get output with transparency report
logits, report = monitor.monitored_forward(input_ids)

print(report)
# {'status': 'ok', 'max_divergence': 0.12, 'mean_divergence': 0.08, ...}
```

## Adaptive Computation

The model learns when to "think more" based on confidence:

```
Token: "2"  → High confidence → Exit at depth 2
Token: "+"  → High confidence → Exit at depth 2  
Token: "2"  → High confidence → Exit at depth 2
Token: "="  → Medium confidence → Exit at depth 4
Token: "?"  → Low confidence → Full depth 8 (needs more compute)
```

### How It Works

```python
class AdaptiveCocoNODE:
    def forward(self, x):
        for block_idx, block in enumerate(self.blocks):
            x = block(x)  # ODE integration
            
            # Compute confidence for each token
            confidence = self.confidence_estimator(x)
            
            # Exit tokens that are confident enough
            # CRITICAL: Add noise to threshold (anti-scheming)
            threshold = self.base_threshold + random.gauss(0, noise_std)
            should_exit = confidence > threshold
            
            # Store results for exited tokens
            results[should_exit] = x[should_exit]
```

### Confidence Metrics

| Metric | Description | Best For |
|--------|-------------|----------|
| `softmax` | max(softmax(logits)) | General use |
| `entropy` | -Σ p log p (inverted) | Uncertainty-aware |
| `margin` | p₁ - p₂ (top-2 gap) | Classification |
| `classifier` | Learned predictor | Task-specific |
| `cosine` | Similarity to prev state | Convergence detection |

### Safety: Why Noisy Thresholds Matter

Without noise, a scheming model could learn:
> "When I want to hide computation, ensure confidence stays just below threshold"

With noise, the exit point is **unpredictable**:
```python
# Model cannot predict actual threshold
actual_threshold = base_threshold + N(0, σ²)  # Random!
```

This makes gaming exit decisions impossible—the model never knows exactly when it will be forced to exit.

## ODE Step Adaptation

Within each ODE block, step size adapts to state dynamics:

```python
def adaptive_ode_solve(z, dynamics_fn):
    while t < 1.0:
        dz = dynamics_fn(z, t)
        
        # Large change → small steps (more precision)
        # Small change → large steps (efficiency)
        step_scale = 1.0 / (1.0 + dz.norm())
        dt = base_dt * step_scale
        
        z = z + dt * dz
        t = t + dt
        
        # Early exit if converged
        if dz.norm() < threshold:
            break
    return z
```

This provides automatic compute allocation: simple tokens get ~2-3 steps, complex reasoning gets 6-8 steps.

## Key Configuration Options

### Adaptive Depth Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `exit_criterion` | Confidence metric (see above) | 'softmax' |
| `confidence_threshold` | Base exit threshold | 0.85 |
| `threshold_noise_std` | Noise for anti-scheming | 0.08 |
| `per_token_exit` | Allow per-token depth | True |
| `min_depth` | Minimum blocks before exit | 2 |
| `max_depth` | Maximum blocks (hard cap) | 8 |
| `ponder_cost` | Cost per computation step | 0.005 |

### ODE Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_ode_blocks` | Number of ODE blocks (each integrates t=0→1) | 4 |
| `ode_method` | Integration method: 'euler' or 'rk4' | 'euler' |
| `ode_h_max` | Max step size (smaller = more accurate) | 0.25 |
| `time_embd` | Dimension of time embedding | 32 |

### Coconut Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_curriculum_stages` | Number of training stages | 5 |
| `latent_steps_per_stage` | Latent steps added per stage | 2 |
| `stage_iters` | Training iterations per stage | 2000 |
| `max_latent_steps` | Maximum latent reasoning iterations | 8 |

### Transparency Settings

| Parameter | Description | Default |
|-----------|-------------|---------|
| `decode_prob` | Probability of decoding each layer | 0.1 |
| `consistency_weight` | Weight for consistency loss | 0.1 |
| `consistency_type` | Loss type: 'kl', 'mse', 'cross_entropy' | 'kl' |
| `alert_threshold` | KL threshold for transparency alerts | 0.5 |

## How It Connects to the TLR Framework

This implementation is a concrete instantiation of the **Transparent Latent Reasoning** framework:

| TLR Component | CocoNODE Implementation |
|---------------|------------------------|
| Stochastic layer decoding | `StochasticDecoder.should_decode()` |
| Variable-depth computation | ODE solver with adaptive stepping |
| Consistency supervision | `ConsistencyLoss` module |
| Paraphrasing injection | (Planned) Text output transformation |
| Adversarial training | `HidingDetector` + `AdversarialTransparencyTrainer` |

## Limitations and Future Work

**Current limitations:**
- ODE integration is slower than discrete layers (use `ode_h_max=0.5` for speed)
- Transparency overhead ~10-20% during training
- Adversarial hiding detector is simplified (needs counterfactual intervention)

**Planned improvements:**
- Cross-position encoding detection
- Inference-time stochastic decoding (currently training-only by default)
- Integration with GRPO for RL fine-tuning
- Sparse autoencoder analysis of transparent representations

## Citation

If you use CocoNODE, please cite:

```bibtex
@misc{coconode2025,
  title={CocoNODE: Neural ODE GPT with Transparent Latent Reasoning},
  author={...},
  year={2025},
  note={Combining continuous depth, continuous thought, and stochastic transparency}
}
```

## Related Work

- [Coconut](https://github.com/facebookresearch/coconut) - Chain of Continuous Thought
- [TRM](https://arxiv.org/abs/2510.04871) - Tiny Recursive Models
- [Neural ODEs](https://arxiv.org/abs/1806.07366) - Original Neural ODE paper
- [Tuned Lens](https://arxiv.org/abs/2303.08112) - Interpreting intermediate layers
- [Apollo Research](https://arxiv.org/abs/2412.04984) - Frontier model scheming