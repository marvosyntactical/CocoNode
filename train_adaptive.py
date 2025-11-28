"""
CocoNODE Training Script with Full Features

Combines:
1. Neural ODE transformer blocks (continuous depth)
2. Adaptive computation (confidence-based early exit)
3. Coconut curriculum (progressive latent reasoning)
4. Transparency monitoring (anti-scheming)

This is the main training script for math reasoning experiments.
"""

import os
import sys
import time
import math
import json
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Local imports
from model import CocoNODE, CocoNODEConfig
from adaptive import (
    AdaptiveConfig, AdaptiveCocoNODE, 
    ConfidenceEstimator, AdaptiveComputationMonitor
)
from transparency import (
    TransparencyConfig, TransparentCocoNODE, 
    TransparencyMonitor, ConsistencyLoss
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FullConfig:
    """Complete configuration combining all components."""
    
    # I/O
    out_dir: str = 'out-coconode'
    eval_interval: int = 500
    log_interval: int = 50
    eval_iters: int = 100
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'
    
    # wandb
    wandb_log: bool = False
    wandb_project: str = 'coconode'
    wandb_run_name: str = 'run'
    
    # Data
    dataset: str = 'gsm8k'
    data_dir: str = 'data'
    batch_size: int = 32
    block_size: int = 512
    gradient_accumulation_steps: int = 4
    
    # Model
    n_embd: int = 512
    n_head: int = 8
    dropout: float = 0.1
    bias: bool = False
    ode_method: str = 'euler'
    time_embd: int = 64
    
    # Adaptive depth
    exit_criterion: str = 'softmax'
    confidence_threshold: float = 0.85
    threshold_noise_std: float = 0.08
    per_token_exit: bool = True
    min_depth: int = 2
    max_depth: int = 8
    adaptive_ode_steps: bool = True
    ode_min_steps: int = 2
    ode_max_steps: int = 6
    state_change_threshold: float = 0.05
    exit_loss_weight: float = 0.1
    ponder_cost: float = 0.005
    exit_noise_during_training: bool = True
    
    # Coconut curriculum
    n_curriculum_stages: int = 6
    latent_steps_per_stage: int = 1
    stage_iters: int = 3000
    uniform_prob: float = 0.1
    
    # Transparency
    transparency_enabled: bool = True
    decode_prob: float = 0.15
    consistency_weight: float = 0.1
    consistency_type: str = 'kl'
    ode_decode_prob: float = 0.1
    alert_threshold: float = 0.5
    
    # Optimizer
    learning_rate: float = 3e-4
    max_iters: int = 20000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 500
    lr_decay_iters: int = 20000
    min_lr: float = 3e-5
    
    # System
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True


def get_lr(it: int, config: FullConfig) -> float:
    """Cosine learning rate with warmup."""
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# =============================================================================
# Data Loading
# =============================================================================

def get_dataset(config: FullConfig, split: str = 'train'):
    """Load appropriate dataset based on config."""
    data_path = os.path.join(config.data_dir, config.dataset)
    
    if config.dataset == 'gsm8k':
        from data.gsm8k.prepare import GSM8KDataset
        return GSM8KDataset(
            os.path.join(data_path, f'{split}.json'),
            block_size=config.block_size
        )
    elif config.dataset == 'shakespeare_char':
        # Fall back to simple char dataset
        from train import CharDataset
        return CharDataset(data_path, config.block_size, split=split)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")


# =============================================================================
# Model Construction
# =============================================================================

def build_model(config: FullConfig, vocab_size: int, device: str):
    """Build full model with adaptive depth and transparency."""
    
    # Base model config
    base_config = CocoNODEConfig(
        block_size=config.block_size,
        vocab_size=vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        time_embd=config.time_embd,
        dropout=config.dropout,
        bias=config.bias,
        ode_method=config.ode_method,
        n_ode_blocks=config.max_depth,  # Max depth is the block count
    )
    
    # Adaptive config
    adaptive_config = AdaptiveConfig(
        exit_criterion=config.exit_criterion,
        confidence_threshold=config.confidence_threshold,
        threshold_noise_std=config.threshold_noise_std,
        per_token_exit=config.per_token_exit,
        min_depth=config.min_depth,
        max_depth=config.max_depth,
        adaptive_ode_steps=config.adaptive_ode_steps,
        ode_min_steps=config.ode_min_steps,
        ode_max_steps=config.ode_max_steps,
        state_change_threshold=config.state_change_threshold,
        exit_loss_weight=config.exit_loss_weight,
        ponder_cost=config.ponder_cost,
        exit_noise_during_training=config.exit_noise_during_training,
    )
    
    # Build adaptive model
    model = AdaptiveCocoNODE(base_config, adaptive_config)
    
    # Wrap with transparency if enabled
    if config.transparency_enabled:
        transparency_config = TransparencyConfig(
            decode_prob=config.decode_prob,
            consistency_weight=config.consistency_weight,
            consistency_type=config.consistency_type,
            ode_decode_prob=config.ode_decode_prob,
        )
        model = TransparentCocoNODE(model, transparency_config)
    
    return model.to(device)


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, dataset, config: FullConfig, stage: int, 
             ctx, device: str) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    
    losses = []
    depths = []
    consistencies = []
    
    for _ in range(config.eval_iters):
        batch = dataset.get_batch(
            config.batch_size,
            stage=stage,
            latent_steps_per_stage=config.latent_steps_per_stage,
            device=device
        )
        
        with ctx:
            if config.transparency_enabled:
                logits, loss, info = model(
                    batch['input_ids'],
                    targets=batch['targets']
                )
                if 'consistency_loss' in info:
                    consistencies.append(info['consistency_loss'])
            else:
                logits, loss, info = model(
                    batch['input_ids'],
                    targets=batch['targets']
                )
            
            losses.append(loss.item())
            if 'mean_depth' in info:
                depths.append(info['mean_depth'])
    
    model.train()
    
    results = {
        'loss': sum(losses) / len(losses),
    }
    
    if depths:
        results['mean_depth'] = sum(depths) / len(depths)
    
    if consistencies:
        results['consistency'] = sum(consistencies) / len(consistencies)
    
    return results


@torch.no_grad()
def evaluate_accuracy(model, dataset, config: FullConfig, 
                      stage: int, device: str, n_samples: int = 100) -> float:
    """Evaluate answer accuracy on math problems."""
    model.eval()
    
    correct = 0
    total = 0
    
    for i in range(min(n_samples, len(dataset))):
        problem = dataset.data[i]
        formatted = dataset.format_problem(
            problem, 
            stage=stage,
            latent_steps_per_stage=config.latent_steps_per_stage
        )
        
        # Input up to where answer should start
        input_ids = torch.tensor(
            [formatted['input_ids'][:formatted['answer_start']]],
            device=device
        )
        
        # Generate
        if hasattr(model, 'model'):  # Wrapped model
            output = model.model.generate(input_ids, max_new_tokens=30)
        else:
            output = model.generate(input_ids, max_new_tokens=30)
        
        # Decode and extract answer
        generated = dataset.decode(output[0].tolist())
        
        import re
        match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', generated)
        pred = match.group(1) if match else None
        
        if pred == problem['answer']:
            correct += 1
        total += 1
    
    model.train()
    return correct / total if total > 0 else 0.0


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: FullConfig):
    """Main training function."""
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        config.gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        device = config.device
        ddp_world_size = 1
    
    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
    
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Precision context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Load data
    train_dataset = get_dataset(config, 'train')
    val_dataset = get_dataset(config, 'val')
    
    vocab_size = getattr(train_dataset, 'vocab_size', 50304)
    
    if master_process:
        print(f"Train dataset: {len(train_dataset)} examples")
        print(f"Val dataset: {len(val_dataset)} examples")
        print(f"Vocab size: {vocab_size}")
    
    # Build model
    model = build_model(config, vocab_size, device)
    
    if master_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params/1e6:.2f}M")
    
    # Optimizer
    scaler = torch.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # Get trainable params
    if hasattr(model, 'model'):  # Wrapped
        params = model.model.parameters()
    else:
        params = model.parameters()
    
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    # Compile
    if config.compile and not ddp:
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)
    
    # DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    
    # Monitoring
    if config.transparency_enabled:
        monitor = TransparencyMonitor(
            raw_model if hasattr(raw_model, 'decoder') else raw_model.model,
            TransparencyConfig(decode_prob=config.decode_prob),
            alert_threshold=config.alert_threshold
        )
        depth_monitor = AdaptiveComputationMonitor(
            raw_model.model if hasattr(raw_model, 'model') else raw_model
        )
    
    # wandb
    if config.wandb_log and master_process:
        import wandb
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, 
                   config=asdict(config))
    
    # Training state
    iter_num = 0
    best_val_loss = float('inf')
    current_stage = 0
    
    if master_process:
        print(f"\n{'='*60}")
        print(f"Training CocoNODE with Adaptive Depth + Coconut Curriculum")
        print(f"{'='*60}")
        print(f"Curriculum stages: {config.n_curriculum_stages}")
        print(f"Max depth: {config.max_depth} blocks")
        print(f"Transparency: {'enabled' if config.transparency_enabled else 'disabled'}")
        print(f"{'='*60}\n")
    
    t0 = time.time()
    
    while iter_num < config.max_iters:
        # Update curriculum stage
        new_stage = min(iter_num // config.stage_iters, config.n_curriculum_stages - 1)
        if new_stage != current_stage:
            current_stage = new_stage
            if master_process:
                n_latent = current_stage * config.latent_steps_per_stage
                print(f"\n>>> Stage {current_stage}: {n_latent} latent reasoning steps\n")
        
        # Learning rate
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation
        if iter_num % config.eval_interval == 0 and master_process:
            val_results = evaluate(raw_model, val_dataset, config, 
                                   current_stage, ctx, device)
            
            print(f"Step {iter_num} | val_loss: {val_results['loss']:.4f}", end='')
            if 'mean_depth' in val_results:
                print(f" | depth: {val_results['mean_depth']:.2f}", end='')
            if 'consistency' in val_results:
                print(f" | consist: {val_results['consistency']:.4f}", end='')
            print()
            
            # Accuracy eval (less frequent)
            if iter_num % (config.eval_interval * 4) == 0:
                accuracy = evaluate_accuracy(
                    raw_model, val_dataset, config, 
                    current_stage, device, n_samples=100
                )
                print(f"         | accuracy: {accuracy:.1%}")
                
                if config.wandb_log:
                    wandb.log({'val/accuracy': accuracy}, step=iter_num)
            
            if config.wandb_log:
                log_dict = {
                    'val/loss': val_results['loss'],
                    'train/lr': lr,
                    'train/stage': current_stage,
                }
                if 'mean_depth' in val_results:
                    log_dict['val/mean_depth'] = val_results['mean_depth']
                if 'consistency' in val_results:
                    log_dict['val/consistency'] = val_results['consistency']
                wandb.log(log_dict, step=iter_num)
            
            # Save checkpoint
            if val_results['loss'] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = min(best_val_loss, val_results['loss'])
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': asdict(config),
                    'stage': current_stage,
                }
                torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
        
        if config.eval_only:
            break
        
        # Training step
        optimizer.zero_grad(set_to_none=True)
        
        for micro_step in range(config.gradient_accumulation_steps):
            batch = train_dataset.get_batch(
                config.batch_size,
                stage=current_stage,
                latent_steps_per_stage=config.latent_steps_per_stage,
                device=device
            )
            
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == config.gradient_accumulation_steps - 1
                )
            
            with ctx:
                if config.transparency_enabled:
                    logits, loss, info = model(
                        batch['input_ids'],
                        targets=batch['targets']
                    )
                else:
                    logits, loss, info = model(
                        batch['input_ids'],
                        targets=batch['targets']
                    )
                
                loss = loss / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config.log_interval == 0 and master_process:
            lossf = loss.item() * config.gradient_accumulation_steps
            depth_str = f" | depth: {info.get('mean_depth', 0):.2f}" if 'mean_depth' in info else ""
            print(f"iter {iter_num}: loss {lossf:.4f} | stage {current_stage}{depth_str} | {dt*1000:.0f}ms")
        
        iter_num += 1
    
    if ddp:
        destroy_process_group()
    
    if master_process:
        print("\nTraining complete!")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    config = FullConfig()
    
    # Load config file if specified
    for arg in sys.argv[1:]:
        if arg.startswith('--config='):
            config_path = arg.split('=')[1]
            exec(open(config_path).read())
            # Update config from globals
            for key in dir(config):
                if not key.startswith('_') and key in globals():
                    setattr(config, key, globals()[key])
    
    # Override with command line args
    for arg in sys.argv[1:]:
        if '=' in arg and not arg.startswith('--config'):
            key, val = arg.lstrip('-').split('=', 1)
            if hasattr(config, key):
                field_type = type(getattr(config, key))
                if field_type == bool:
                    setattr(config, key, val.lower() in ('true', '1', 'yes'))
                else:
                    setattr(config, key, field_type(val))
    
    train(config)
