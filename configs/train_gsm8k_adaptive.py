"""
Configuration for training CocoNODE on GSM8K with:
- Adaptive depth (confidence-based early exit)
- Coconut curriculum (progressive latent reasoning)
- Transparency monitoring (anti-scheming)

This is the full-featured config for math reasoning experiments.

Usage:
    # Prepare data first
    python data/gsm8k/prepare.py --output_dir data/gsm8k
    
    # Train
    python train_adaptive.py --config configs/train_gsm8k_adaptive.py
    
    # Train with monitoring
    python train_adaptive.py --config configs/train_gsm8k_adaptive.py \
        --transparency_enabled=True \
        --wandb_log=True
"""

# =============================================================================
# I/O
# =============================================================================
out_dir = 'out-coconode-gsm8k'
eval_interval = 500
log_interval = 50
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch', 'resume', or path to checkpoint

# wandb logging
wandb_log = True
wandb_project = 'coconode-gsm8k'
wandb_run_name = 'adaptive-coconut'

# =============================================================================
# Data
# =============================================================================
dataset = 'gsm8k'
data_dir = 'data'
batch_size = 32
block_size = 512  # GSM8K needs longer context than Shakespeare
gradient_accumulation_steps = 4

# =============================================================================
# Model Architecture
# =============================================================================
n_embd = 512
n_head = 8
dropout = 0.1
bias = False

# ODE settings
ode_method = 'euler'
time_embd = 64  # Larger time embedding for more expressive dynamics

# =============================================================================
# Adaptive Depth Settings
# =============================================================================
# These control confidence-based early exit

# Exit criterion: 'softmax', 'entropy', 'margin', 'classifier', 'cosine'
exit_criterion = 'softmax'

# Confidence threshold for exiting
confidence_threshold = 0.85

# Add noise to threshold (CRITICAL for anti-scheming)
# This makes exit points unpredictable to the model
threshold_noise_std = 0.08

# Allow different tokens to exit at different depths
per_token_exit = True

# Depth bounds
min_depth = 2   # At least 2 ODE blocks before exit allowed
max_depth = 8   # Maximum 8 blocks (hard cap)

# ODE step adaptation within blocks
adaptive_ode_steps = True
ode_min_steps = 2
ode_max_steps = 6
state_change_threshold = 0.05  # Exit ODE early if state barely changing

# Loss weights
exit_loss_weight = 0.1   # Weight for training exit heads
ponder_cost = 0.005      # Small cost per computation step (encourages efficiency)

# =============================================================================
# Coconut Curriculum Settings
# =============================================================================
# Progressive replacement of CoT steps with latent tokens

n_curriculum_stages = 6        # Number of training stages
latent_steps_per_stage = 1     # Replace 1 more step per stage
stage_iters = 3000             # Iterations per stage (18K total)
uniform_prob = 0.1             # 10% chance to sample from different stage

# At stage k, first k reasoning steps become latent tokens
# Stage 0: Full CoT text
# Stage 1: Step 1 is latent, rest is text
# Stage 5: Steps 1-5 are latent, rest is text

# =============================================================================
# Optimizer
# =============================================================================
learning_rate = 3e-4
max_iters = 20000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 500
lr_decay_iters = 20000
min_lr = 3e-5

# =============================================================================
# Transparency / Anti-Scheming Settings
# =============================================================================
transparency_enabled = True

# Stochastic decoding probability
decode_prob = 0.15  # Decode ~15% of layers randomly

# Consistency loss weight
consistency_weight = 0.1

# Consistency loss type: 'kl', 'mse', 'cross_entropy'
consistency_type = 'kl'

# ODE-specific decoding (decode at random integration steps)
ode_decode_prob = 0.1

# Alert threshold for monitoring
alert_threshold = 0.5

# Add noise to exits during training (prevents learning to game exits)
exit_noise_during_training = True

# =============================================================================
# System
# =============================================================================
device = 'cuda'
dtype = 'bfloat16'
compile = True

# =============================================================================
# Evaluation Settings
# =============================================================================
eval_samples = 200           # Number of problems for evaluation
eval_at_curriculum_end = True  # Full eval at end of each curriculum stage


# =============================================================================
# Experiment Tracking
# =============================================================================
# These will be logged to wandb for analysis

track_metrics = [
    'train_loss',
    'val_loss', 
    'val_accuracy',
    'mean_depth',
    'depth_std',
    'exit_distribution',
    'consistency_loss',
    'ponder_cost',
    'curriculum_stage',
    'n_latent_steps',
]

# Save detailed exit patterns for safety analysis
save_exit_patterns = True
exit_pattern_interval = 1000  # Save every N iterations
