"""
CocoNODE Training Script

Implements Coconut-style curriculum training for the Neural ODE GPT:
1. Stage 0: Standard language modeling (no latent reasoning)
2. Stage k: Replace k reasoning steps with continuous thoughts

Key features:
- Multi-stage curriculum learning
- Stochastic intermediate decoding for interpretability
- Deep supervision at random ODE integration points
- Compatible with DDP for multi-GPU training
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import CocoNODE, CocoNODEConfig


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    # I/O
    out_dir: str = 'out-coconode'
    eval_interval: int = 500
    log_interval: int = 10
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = 'scratch'  # 'scratch' or 'resume' or 'gpt2'
    
    # wandb logging
    wandb_log: bool = False
    wandb_project: str = 'coconode'
    wandb_run_name: str = 'run'
    
    # Data
    dataset: str = 'gsm8k'  # 'gsm8k', 'prontoqa', 'shakespeare_char'
    data_dir: str = 'data'
    gradient_accumulation_steps: int = 8
    batch_size: int = 12
    block_size: int = 256
    
    # Model
    n_embd: int = 384
    n_head: int = 6
    n_ode_blocks: int = 4
    time_embd: int = 32
    dropout: float = 0.1
    bias: bool = False
    ode_method: str = 'euler'
    ode_h_max: float = 0.25
    
    # Coconut curriculum
    n_curriculum_stages: int = 5  # Number of training stages
    latent_steps_per_stage: int = 2  # Latent steps to add per stage
    stage_iters: int = 2000  # Iterations per curriculum stage
    uniform_prob: float = 0.0  # Probability to mix data from other stages
    
    # Optimizer
    learning_rate: float = 1e-3
    max_iters: int = 10000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Learning rate schedule
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 10000
    min_lr: float = 1e-4
    
    # System
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    compile: bool = True


# =============================================================================
# Data Loading
# =============================================================================

class CoconutDataset:
    """
    Dataset for Coconut-style training with reasoning steps.
    
    Each example has:
    - question: The input question
    - answer: The final answer
    - steps: List of intermediate reasoning steps
    
    During training, we progressively replace language steps with latent tokens.
    """
    
    def __init__(self, data_path: str, block_size: int, 
                 vocab_size: int = 50304, split: str = 'train'):
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Simple character-level tokenization for now
        # In practice, use tiktoken or similar
        self.stoi = {chr(i): i for i in range(256)}
        self.itos = {i: chr(i) for i in range(256)}
        
        # Special tokens
        self.bot_token = 256  # Begin of thought
        self.eot_token = 257  # End of thought
        self.pad_token = 258
        
    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(c, 0) for c in s]
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join(self.itos.get(t, '?') for t in tokens if t < 256)
    
    def get_batch(self, batch_size: int, stage: int = 0, 
                  device: str = 'cuda') -> Dict[str, Any]:
        """
        Get a batch for the given curriculum stage.
        
        At stage k, replace k reasoning steps with latent positions.
        """
        indices = np.random.randint(0, len(self.data), size=batch_size)
        
        batch_tokens = []
        batch_targets = []
        batch_latent_positions = []
        
        for idx in indices:
            example = self.data[idx]
            
            # Construct sequence: question + steps + answer
            question_tokens = self.encode(example.get('question', ''))
            answer_tokens = self.encode(example.get('answer', ''))
            steps = example.get('steps', [])
            
            # Build sequence with latent positions for this stage
            tokens = question_tokens.copy()
            latent_positions = []
            
            for step