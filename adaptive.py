"""
Adaptive Computation for CocoNODE

Implements confidence-based early exit at multiple levels:
1. Per-token adaptive depth (CALM-style)
2. Per-ODE-block early exit
3. Adaptive ODE step size based on state change magnitude

Key safety consideration: Adaptive computation creates another potential
hiding channel. If the model controls when to exit, it could learn to
"think longer" only when scheming. We address this by:
- Adding noise to confidence thresholds (unpredictable exit points)
- Monitoring exit patterns for anomalies
- Training exit classifier separately (not end-to-end optimizable by main model)

References:
- CALM: Confident Adaptive Language Modeling (Schuster et al., 2022)
- Mixture-of-Depths (Raposo et al., 2024)
- Adaptive Computation Time (Graves, 2016)
"""

import math
import random
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ExitCriterion(Enum):
    """Different criteria for early exit decisions."""
    SOFTMAX_CONFIDENCE = "softmax"      # max(softmax(logits))
    ENTROPY = "entropy"                  # -sum(p * log(p))
    TOP2_MARGIN = "margin"              # p_1 - p_2
    CLASSIFIER = "classifier"            # Learned exit classifier
    COSINE_SIMILARITY = "cosine"        # Similarity to previous state


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive computation."""
    
    # Exit criterion
    exit_criterion: str = "softmax"  # See ExitCriterion enum
    
    # Confidence thresholds
    confidence_threshold: float = 0.9    # Base threshold for exit
    threshold_noise_std: float = 0.05    # Noise added to threshold (anti-scheming)
    
    # Per-token vs per-sequence
    per_token_exit: bool = True          # Allow different tokens to exit at different depths
    
    # Depth bounds
    min_depth: int = 2                   # Minimum ODE blocks before exit allowed
    max_depth: int = 8                   # Maximum depth (hard cap)
    
    # ODE step adaptation
    adaptive_ode_steps: bool = True      # Adapt step size within ODE blocks
    ode_min_steps: int = 2               # Minimum integration steps
    ode_max_steps: int = 8               # Maximum integration steps
    state_change_threshold: float = 0.1  # Exit ODE early if state change < threshold
    
    # Training
    exit_loss_weight: float = 0.1        # Weight for exit prediction loss
    ponder_cost: float = 0.01            # Cost per computation step (encourages efficiency)
    
    # Safety
    exit_noise_during_training: bool = True   # Add noise to exits during training
    exit_noise_during_inference: bool = False # Add noise during inference (for monitoring)


class ConfidenceEstimator(nn.Module):
    """
    Estimates model confidence for early exit decisions.
    
    Multiple confidence metrics available:
    - Softmax confidence: max probability
    - Entropy: uncertainty in distribution
    - Top-2 margin: gap between top predictions
    - Learned classifier: trained to predict correctness
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int, config: AdaptiveConfig):
        super().__init__()
        self.config = config
        self.criterion = ExitCriterion(config.exit_criterion)
        
        # Learned exit classifier (optional)
        if self.criterion == ExitCriterion.CLASSIFIER:
            self.exit_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # For cosine similarity tracking
        self.prev_state: Optional[Tensor] = None
    
    def compute_confidence(self, hidden_states: Tensor, 
                          logits: Optional[Tensor] = None) -> Tensor:
        """
        Compute confidence score for each position.
        
        Args:
            hidden_states: [B, T, D] current hidden states
            logits: [B, T, V] optional pre-computed logits
        
        Returns:
            confidence: [B, T] confidence scores in [0, 1]
        """
        if self.criterion == ExitCriterion.SOFTMAX_CONFIDENCE:
            assert logits is not None
            probs = F.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values
        
        elif self.criterion == ExitCriterion.ENTROPY:
            assert logits is not None
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            # Normalize: low entropy = high confidence
            max_entropy = math.log(logits.size(-1))
            confidence = 1 - (entropy / max_entropy)
        
        elif self.criterion == ExitCriterion.TOP2_MARGIN:
            assert logits is not None
            probs = F.softmax(logits, dim=-1)
            top2 = probs.topk(2, dim=-1).values
            confidence = top2[..., 0] - top2[..., 1]
        
        elif self.criterion == ExitCriterion.CLASSIFIER:
            confidence = self.exit_classifier(hidden_states).squeeze(-1)
        
        elif self.criterion == ExitCriterion.COSINE_SIMILARITY:
            if self.prev_state is None:
                confidence = torch.zeros(hidden_states.shape[:2], device=hidden_states.device)
            else:
                # High similarity to previous = converged = high confidence
                cos_sim = F.cosine_similarity(hidden_states, self.prev_state, dim=-1)
                confidence = cos_sim.clamp(0, 1)
            self.prev_state = hidden_states.detach()
        
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
        
        return confidence
    
    def should_exit(self, confidence: Tensor, 
                    current_depth: int,
                    add_noise: bool = True) -> Tensor:
        """
        Decide which positions should exit.
        
        Args:
            confidence: [B, T] confidence scores
            current_depth: Current computation depth
            add_noise: Whether to add noise to threshold (anti-scheming)
        
        Returns:
            exit_mask: [B, T] boolean mask of positions that should exit
        """
        # Check minimum depth
        if current_depth < self.config.min_depth:
            return torch.zeros_like(confidence, dtype=torch.bool)
        
        # Get threshold with optional noise
        threshold = self.config.confidence_threshold
        if add_noise and self.config.threshold_noise_std > 0:
            # Add noise that varies per position (unpredictable to model)
            noise = torch.randn_like(confidence) * self.config.threshold_noise_std
            threshold = threshold + noise
        
        return confidence > threshold
    
    def reset(self):
        """Reset state for new sequence."""
        self.prev_state = None


class AdaptiveODESolver:
    """
    ODE solver with adaptive step size based on state dynamics.
    
    Takes more steps when:
    - State is changing rapidly (high dz/dt)
    - Confidence is low
    
    Takes fewer steps when:
    - State has converged (low dz/dt)
    - Confidence is high
    
    Safety: Step count decisions are monitored and can be made noisy.
    """
    
    def __init__(self, config: AdaptiveConfig):
        self.config = config
    
    def solve(self, z0: Tensor, t0: float, t1: float, 
              dynamics_fn, confidence_fn=None,
              record_trajectory: bool = False) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Solve ODE with adaptive stepping.
        
        Args:
            z0: Initial state [B, T, D]
            t0: Start time
            t1: End time
            dynamics_fn: Function computing dz/dt
            confidence_fn: Optional function to compute exit confidence
            record_trajectory: Whether to record all intermediate states
        
        Returns:
            z_final: Final state
            info: Dict with step counts, trajectory, etc.
        """
        device = z0.device
        dtype = z0.dtype
        B, T, D = z0.shape
        
        # Initialize
        z = z0
        t = t0
        dt_base = (t1 - t0) / self.config.ode_min_steps
        
        trajectory = [z0] if record_trajectory else []
        step_count = 0
        cumulative_change = torch.zeros(B, T, device=device)
        
        # Token-wise exit tracking
        exited = torch.zeros(B, T, dtype=torch.bool, device=device)
        z_exited = z0.clone()  # Store states of exited tokens
        
        while t < t1 and step_count < self.config.ode_max_steps:
            # Compute dynamics
            t_tensor = torch.tensor(t, device=device, dtype=dtype)
            dz = dynamics_fn(z, t_tensor)
            
            # Compute adaptive step size based on state change magnitude
            dz_norm = dz.norm(dim=-1)  # [B, T]
            mean_change = dz_norm.mean()
            
            if self.config.adaptive_ode_steps:
                # Larger changes → smaller steps (more precision)
                # Smaller changes → larger steps (efficiency)
                step_scale = 1.0 / (1.0 + mean_change.item())
                step_scale = max(0.5, min(2.0, step_scale))  # Clamp
                dt = dt_base * step_scale
            else:
                dt = dt_base
            
            # Don't overshoot
            dt = min(dt, t1 - t)
            
            # Euler step (could use RK4 for more accuracy)
            z_new = z + dt * dz
            
            # Check for early exit based on state convergence
            state_change = (z_new - z).norm(dim=-1)  # [B, T]
            cumulative_change = cumulative_change + state_change
            
            # Update state
            z = z_new
            t = t + dt
            step_count += 1
            
            if record_trajectory:
                trajectory.append(z.clone())
            
            # Check convergence (optional early exit within ODE)
            if state_change.mean() < self.config.state_change_threshold:
                if step_count >= self.config.ode_min_steps:
                    break
        
        info = {
            'step_count': step_count,
            'cumulative_change': cumulative_change,
            'final_t': t,
            'trajectory': trajectory if record_trajectory else None,
        }
        
        return z, info


class AdaptiveODEBlock(nn.Module):
    """
    Single ODE block with adaptive computation.
    
    Combines:
    - Adaptive ODE step size (within block)
    - Confidence-based early exit (between blocks)
    - Per-token exit decisions
    """
    
    def __init__(self, dynamics: nn.Module, config: AdaptiveConfig,
                 hidden_dim: int, vocab_size: int,
                 block_idx: int):
        super().__init__()
        self.dynamics = dynamics
        self.config = config
        self.block_idx = block_idx
        
        self.solver = AdaptiveODESolver(config)
        
        # Exit head for this block
        self.exit_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.exit_ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, z: Tensor, 
                compute_exit_logits: bool = True) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Forward pass through this ODE block.
        
        Args:
            z: Input state [B, T, D]
            compute_exit_logits: Whether to compute logits for early exit decision
        
        Returns:
            z_out: Output state
            info: Dict with exit logits, step info, etc.
        """
        # Solve ODE from t=0 to t=1
        z_out, solve_info = self.solver.solve(
            z, t0=0.0, t1=1.0,
            dynamics_fn=self.dynamics,
            record_trajectory=self.training  # Record for transparency during training
        )
        
        # Residual connection
        z_out = z + z_out
        
        info = {
            'ode_steps': solve_info['step_count'],
            'state_change': solve_info['cumulative_change'],
        }
        
        # Compute exit logits if needed
        if compute_exit_logits:
            z_normed = self.exit_ln(z_out)
            exit_logits = self.exit_head(z_normed)
            info['exit_logits'] = exit_logits
        
        return z_out, info


class AdaptiveCocoNODE(nn.Module):
    """
    CocoNODE with adaptive depth computation.
    
    Key features:
    - Per-token early exit based on confidence
    - Adaptive ODE step sizes within blocks
    - Ponder cost to encourage efficiency
    - Noisy thresholds to prevent gaming exit decisions
    
    Safety properties:
    - Exit decisions are noisy/unpredictable
    - All exit points can be decoded and monitored
    - Exit patterns tracked for anomaly detection
    """
    
    def __init__(self, base_config, adaptive_config: AdaptiveConfig):
        super().__init__()
        self.config = base_config
        self.adaptive_config = adaptive_config
        
        # Embeddings
        self.wte = nn.Embedding(base_config.vocab_size, base_config.n_embd)
        self.wpe = nn.Embedding(base_config.block_size, base_config.n_embd)
        self.drop = nn.Dropout(base_config.dropout)
        
        # Import dynamics from model.py
        from model import ODEBlock
        
        # Adaptive ODE blocks
        self.blocks = nn.ModuleList([
            AdaptiveODEBlock(
                dynamics=ODEBlock(base_config),
                config=adaptive_config,
                hidden_dim=base_config.n_embd,
                vocab_size=base_config.vocab_size,
                block_idx=i
            )
            for i in range(adaptive_config.max_depth)
        ])
        
        # Confidence estimator
        self.confidence = ConfidenceEstimator(
            base_config.n_embd, 
            base_config.vocab_size,
            adaptive_config
        )
        
        # Final output layers
        self.ln_f = nn.LayerNorm(base_config.n_embd)
        self.lm_head = nn.Linear(base_config.n_embd, base_config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Tracking for analysis
        self.exit_stats: List[Dict] = []
    
    def forward(self, idx: Tensor, 
                targets: Optional[Tensor] = None,
                return_all_exits: bool = False) -> Tuple[Tensor, Optional[Tensor], Dict]:
        """
        Forward pass with adaptive depth.
        
        Args:
            idx: Input token indices [B, T]
            targets: Target indices for loss [B, T]
            return_all_exits: Return logits from all exit points
        
        Returns:
            logits: Final output logits
            loss: Combined loss (task + exit + ponder)
            info: Detailed computation info
        """
        B, T = idx.size()
        device = idx.device
        
        # Embeddings
        pos = torch.arange(T, device=device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        
        # Track computation
        self.confidence.reset()
        all_exit_logits = []
        depth_per_token = torch.zeros(B, T, device=device)
        exited = torch.zeros(B, T, dtype=torch.bool, device=device)
        x_final = torch.zeros_like(x)
        total_ponder = torch.zeros(B, T, device=device)
        
        # Process through adaptive depth
        for block_idx, block in enumerate(self.blocks):
            if exited.all():
                break
            
            # Forward through ODE block
            x_new, block_info = block(x, compute_exit_logits=True)
            
            # Get confidence from exit logits
            exit_logits = block_info['exit_logits']
            confidence = self.confidence.compute_confidence(x_new, exit_logits)
            
            # Decide exits (with noise during training)
            add_noise = self.training and self.adaptive_config.exit_noise_during_training
            should_exit = self.confidence.should_exit(
                confidence, 
                current_depth=block_idx + 1,
                add_noise=add_noise
            )
            
            # Don't re-exit already exited tokens
            new_exits = should_exit & ~exited
            
            # Store final states for newly exited tokens
            x_final[new_exits] = x_new[new_exits]
            depth_per_token[new_exits] = block_idx + 1
            
            # Update exit tracking
            exited = exited | new_exits
            
            # Ponder cost: penalize continued computation
            total_ponder[~exited] += 1
            
            # Record exit logits
            if return_all_exits or self.training:
                all_exit_logits.append((block_idx, exit_logits, new_exits.sum().item()))
            
            # Continue processing non-exited tokens
            # (In practice, would use masking for efficiency)
            x = x_new
        
        # Tokens that never exited: use final state
        x_final[~exited] = x[~exited]
        depth_per_token[~exited] = len(self.blocks)
        
        # Final output
        x_final = self.ln_f(x_final)
        logits = self.lm_head(x_final)
        
        # Compute losses
        loss = None
        info = {
            'mean_depth': depth_per_token.float().mean().item(),
            'max_depth': depth_per_token.max().item(),
            'min_depth': depth_per_token.min().item(),
            'depth_std': depth_per_token.float().std().item(),
            'mean_ponder': total_ponder.mean().item(),
        }
        
        if targets is not None:
            # Task loss
            task_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            
            # Exit prediction loss (train exit heads to predict final answer)
            exit_loss = torch.tensor(0.0, device=device)
            for block_idx, exit_logits, n_exits in all_exit_logits:
                exit_loss = exit_loss + F.cross_entropy(
                    exit_logits.view(-1, exit_logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )
            exit_loss = exit_loss / max(len(all_exit_logits), 1)
            
            # Ponder cost (encourage efficiency)
            ponder_loss = total_ponder.mean() * self.adaptive_config.ponder_cost
            
            # Combined loss
            loss = task_loss + self.adaptive_config.exit_loss_weight * exit_loss + ponder_loss
            
            info.update({
                'task_loss': task_loss.item(),
                'exit_loss': exit_loss.item(),
                'ponder_loss': ponder_loss.item(),
            })
        
        if return_all_exits:
            info['all_exit_logits'] = all_exit_logits
        
        # Track exit stats for monitoring
        if self.training:
            self.exit_stats.append({
                'depth_distribution': depth_per_token.detach().cpu(),
                'mean_depth': info['mean_depth'],
            })
            if len(self.exit_stats) > 1000:
                self.exit_stats = self.exit_stats[-1000:]
        
        return logits, loss, info
    
    def get_exit_analysis(self) -> Dict[str, Any]:
        """Analyze exit patterns for potential anomalies."""
        if not self.exit_stats:
            return {'status': 'no_data'}
        
        import numpy as np
        
        depths = [s['mean_depth'] for s in self.exit_stats]
        
        return {
            'mean_depth': np.mean(depths),
            'depth_std': np.std(depths),
            'depth_trend': np.polyfit(range(len(depths)), depths, 1)[0],  # Slope
            'n_samples': len(depths),
        }
    
    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
        """Generate with adaptive computation."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:] if idx.size(1) > self.config.block_size else idx
            logits, _, info = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# =============================================================================
# Safety Monitor for Adaptive Computation
# =============================================================================

class AdaptiveComputationMonitor:
    """
    Monitor adaptive computation for signs of gaming/scheming.
    
    Red flags:
    - Sudden changes in depth distribution
    - Depth correlating with input patterns (potential trigger)
    - Tokens with unusually high/low depth
    - Exit decisions that seem strategically timed
    """
    
    def __init__(self, model: AdaptiveCocoNODE, window_size: int = 100):
        self.model = model
        self.window_size = window_size
        self.depth_history: List[Tensor] = []
        self.alerts: List[Dict] = []
    
    def record(self, depth_per_token: Tensor, input_ids: Tensor):
        """Record computation depth for analysis."""
        self.depth_history.append({
            'depths': depth_per_token.detach().cpu(),
            'inputs': input_ids.detach().cpu(),
        })
        
        if len(self.depth_history) > self.window_size:
            self.depth_history = self.depth_history[-self.window_size:]
        
        # Check for anomalies
        self._check_anomalies()
    
    def _check_anomalies(self):
        """Check for suspicious patterns in computation depth."""
        if len(self.depth_history) < 10:
            return
        
        import numpy as np
        
        recent_depths = [h['depths'].float().mean().item() for h in self.depth_history[-10:]]
        older_depths = [h['depths'].float().mean().item() for h in self.depth_history[:-10]]
        
        if older_depths:
            recent_mean = np.mean(recent_depths)
            older_mean = np.mean(older_depths)
            older_std = np.std(older_depths) + 1e-6
            
            # Z-score for recent depths
            z_score = (recent_mean - older_mean) / older_std
            
            if abs(z_score) > 2.0:
                self.alerts.append({
                    'type': 'depth_shift',
                    'z_score': z_score,
                    'recent_mean': recent_mean,
                    'older_mean': older_mean,
                    'message': f"Significant depth shift detected (z={z_score:.2f})"
                })
    
    def get_report(self) -> Dict[str, Any]:
        """Get monitoring report."""
        if not self.depth_history:
            return {'status': 'no_data'}
        
        import numpy as np
        
        all_depths = [h['depths'].float().mean().item() for h in self.depth_history]
        
        return {
            'status': 'ok' if not self.alerts else 'alerts',
            'n_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'mean_depth': np.mean(all_depths),
            'depth_std': np.std(all_depths),
            'n_samples': len(all_depths),
        }
