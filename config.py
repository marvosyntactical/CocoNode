"""
Configuration for training CocoNODE on Shakespeare character-level data.

This is a minimal config for quick experimentation on a single GPU.
For Coconut-style reasoning experiments, use the GSM8K config instead.

Usage:
    python train.py --config configs/train_shakespeare.py

Or override specific settings:
    python train.py --config configs/train_shakespeare.py --n_ode_blocks=6 --ode_h_max=0.1
"""

# I/O
out_dir = 'out-coconode-shakespeare'
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'

# wandb
wandb_log = False
wandb_project = 'coconode'
wandb_run_name = 'shakespeare-char'

# Data
dataset = 'shakespeare_char'
data_dir = 'data'
batch_size = 64
block_size = 256
gradient_accumulation_steps = 1

# Model - Small for quick iteration
n_embd = 384
n_head = 6
n_ode_blocks = 4      # 4 ODE blocks, each integrates t=0->1
time_embd = 32        # Time embedding dimension
dropout = 0.1
bias = False

# ODE settings
ode_method = 'euler'  # 'euler' or 'rk4'
ode_h_max = 0.25      # Max step size (4 steps per block with euler)
                      # Decrease for more accurate integration

# Coconut curriculum (disabled for char-level, no reasoning steps)
n_curriculum_stages = 1
latent_steps_per_stage = 0
stage_iters = 10000
uniform_prob = 0.0

# Optimizer
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4

# System
device = 'cuda'
dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16'
compile = True       # Use PyTorch 2.0 compilation


# =============================================================================
# Transparency settings (for safety experiments)
# =============================================================================

# Stochastic decoding during training
transparency_enabled = True
decode_prob = 0.1           # Decode ~10% of layers randomly
consistency_weight = 0.05   # Weight for consistency loss

# For inference monitoring
inference_decode_prob = 0.2  # Higher for monitoring
alert_threshold = 0.5        # KL divergence threshold for alerts
