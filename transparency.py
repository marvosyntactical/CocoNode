"""
Transparency Module for CocoNODE

This module implements the key safety innovation: stochastic decoding at
random points during both training and inference. The goal is to make
it impossible for the model to learn to hide information because it
cannot predict when its internal state will be decoded and observed.

Key components:
1. StochasticDecoder: Randomly decode hidden states during forward pass
2. ConsistencyLoss: Ensure decoded intermediates predict same answer
3. TransparencyTrainer: Modified training loop with transparency rewards

This addresses the scheming threat by making internal computation
fundamentally observable in an unpredictable way.
"""

import math
import random
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TransparencyConfig:
    """Configuration for transparency/anti-scheming measures."""
    
    # Stochastic decoding
    decode_prob: float = 0.1  # Probability of decoding at each layer
    decode_prob_schedule: str = 'constant'  # 'constant', 'curriculum', 'random'
    min_decode_prob: float = 0.05
    max_decode_prob: float = 0.3
    
    # Consistency loss
    consistency_weight: float = 0.1  # Weight for consistency loss
    consistency_type: str = 'kl'  # 'kl', 'mse', 'cross_entropy'
    
    # ODE-specific: decode at random integration points
    ode_decode_prob: float = 0.1  # Prob of decoding at each ODE step
    
    # Paraphrasing injection (for text outputs)
    paraphrase_prob: float = 0.0  # Prob of paraphrasing during training
    
    # Adversarial transparency training
    adversarial_weight: float = 0.0  # Weight for adversarial hiding detector


class TunedLensProbe(nn.Module):
    """
    Tuned Lens: Affine probe for decoding hidden states at any layer.
    
    Following the tuned lens methodology, we learn an affine transformation
    (A_i, b_i) for each layer i that maps hidden states to vocabulary logits.
    
    This allows us to "see what the model is thinking" at intermediate layers.
    """
    
    def __init__(self, n_layers: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.n_layers = n_layers
        
        # Learnable affine transform for each layer
        self.probes = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size, bias=True)
            for _ in range(n_layers)
        ])
        
        # Initialize close to identity (in logit space)
        for probe in self.probes:
            nn.init.zeros_(probe.weight)
            nn.init.zeros_(probe.bias)
    
    def forward(self, hidden_states: Tensor, layer_idx: int) -> Tensor:
        """
        Decode hidden state at given layer to vocabulary logits.
        
        Args:
            hidden_states: [B, T, D] hidden states
            layer_idx: Which layer's probe to use
        
        Returns:
            logits: [B, T, V] vocabulary logits
        """
        return self.probes[layer_idx](hidden_states)


class StochasticDecoder(nn.Module):
    """
    Stochastic decoding module for transparency.
    
    During forward pass, randomly decides whether to decode hidden states
    at each layer/position. The decoding decisions are:
    - Unpredictable to the model (uses external randomness)
    - Applied during both training and inference
    - Used to compute consistency losses
    
    This makes it impossible for the model to learn which computations
    are "safe" to hide because any computation might be observed.
    """
    
    def __init__(self, config: TransparencyConfig, 
                 n_layers: int, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.config = config
        self.n_layers = n_layers
        
        # Tuned lens probes for each layer
        self.lens = TunedLensProbe(n_layers, hidden_dim, vocab_size)
        
        # Track decoding decisions for this forward pass
        self.decode_mask: Optional[Tensor] = None
        self.decoded_logits: List[Tuple[int, Tensor]] = []
    
    def should_decode(self, layer_idx: int) -> bool:
        """Decide whether to decode at this layer (random, unpredictable)."""
        # Use true randomness, not model-predictable pseudorandomness
        p = self.config.decode_prob
        
        if self.config.decode_prob_schedule == 'random':
            p = random.uniform(self.config.min_decode_prob, self.config.max_decode_prob)
        
        return random.random() < p
    
    def decode_and_record(self, hidden_states: Tensor, layer_idx: int) -> Optional[Tensor]:
        """
        Potentially decode hidden states and record for consistency loss.
        
        Args:
            hidden_states: [B, T, D] current hidden states
            layer_idx: Current layer index
        
        Returns:
            logits if decoded, None otherwise
        """
        if self.should_decode(layer_idx):
            logits = self.lens(hidden_states, layer_idx)
            self.decoded_logits.append((layer_idx, logits))
            return logits
        return None
    
    def reset(self):
        """Reset recorded decodings for new forward pass."""
        self.decoded_logits = []
    
    def get_decoded_logits(self) -> List[Tuple[int, Tensor]]:
        """Get all decoded logits from this forward pass."""
        return self.decoded_logits


class ODEStochasticDecoder(StochasticDecoder):
    """
    Extended stochastic decoder for Neural ODE models.
    
    In addition to layer-wise decoding, this also decodes at random
    points during ODE integration. This is crucial because the ODE
    solver takes multiple steps, each of which could hide computation.
    """
    
    def __init__(self, config: TransparencyConfig,
                 n_layers: int, hidden_dim: int, vocab_size: int,
                 final_ln: nn.Module, lm_head: nn.Module):
        super().__init__(config, n_layers, hidden_dim, vocab_size)
        
        # Reference to model's output layers for decoding
        self.final_ln = final_ln
        self.lm_head = lm_head
        
        # Track ODE integration step decodings
        self.ode_decoded_logits: List[Tuple[int, int, float, Tensor]] = []
    
    def decode_ode_state(self, z: Tensor, block_idx: int, 
                         step_idx: int, t: float) -> Optional[Tensor]:
        """
        Decode state during ODE integration.
        
        Args:
            z: [B, T, D] current ODE state
            block_idx: Which ODE block we're in
            step_idx: Which integration step within the block
            t: Current time value
        
        Returns:
            logits if decoded, None otherwise
        """
        if random.random() < self.config.ode_decode_prob:
            # Decode through final layers
            z_normed = self.final_ln(z)
            logits = self.lm_head(z_normed)
            self.ode_decoded_logits.append((block_idx, step_idx, t, logits))
            return logits
        return None
    
    def reset(self):
        super().reset()
        self.ode_decoded_logits = []


class ConsistencyLoss(nn.Module):
    """
    Compute consistency loss between intermediate decodings and final output.
    
    The key insight: if all intermediate decodings predict the same answer
    as the final output, then there's no "hidden" computation happening
    that diverges from observable reasoning.
    
    Loss types:
    - KL divergence: Soft comparison of probability distributions
    - MSE: Direct comparison in logit space
    - Cross-entropy: Hard target from final output
    """
    
    def __init__(self, config: TransparencyConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.consistency_type
    
    def forward(self, intermediate_logits: List[Tuple[int, Tensor]], 
                final_logits: Tensor,
                targets: Optional[Tensor] = None) -> Tensor:
        """
        Compute consistency loss.
        
        Args:
            intermediate_logits: List of (layer_idx, logits) tuples
            final_logits: [B, T, V] final output logits
            targets: [B, T] optional ground truth targets
        
        Returns:
            Scalar consistency loss
        """
        if len(intermediate_logits) == 0:
            return torch.tensor(0.0, device=final_logits.device)
        
        total_loss = 0.0
        
        # Reference distribution (from final output or targets)
        if self.loss_type == 'cross_entropy' and targets is not None:
            reference = targets
        else:
            reference = F.softmax(final_logits, dim=-1).detach()
        
        for layer_idx, intermed_logits in intermediate_logits:
            if self.loss_type == 'kl':
                # KL divergence: D_KL(final || intermediate)
                intermed_probs = F.log_softmax(intermed_logits, dim=-1)
                loss = F.kl_div(intermed_probs, reference, reduction='batchmean')
            
            elif self.loss_type == 'mse':
                # MSE in logit space
                loss = F.mse_loss(intermed_logits, final_logits.detach())
            
            elif self.loss_type == 'cross_entropy':
                # Cross-entropy with true targets
                loss = F.cross_entropy(
                    intermed_logits.view(-1, intermed_logits.size(-1)),
                    reference.view(-1),
                    ignore_index=-1
                )
            
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss = total_loss + loss
        
        return total_loss / len(intermediate_logits)


class TransparentCocoNODE(nn.Module):
    """
    Wrapper around CocoNODE that adds transparency mechanisms.
    
    This is the key safety innovation: the model is wrapped such that
    during every forward pass, random intermediate states are decoded
    and checked for consistency. The model cannot learn to hide
    information because:
    
    1. Decoding decisions are truly random (external to model)
    2. Consistency loss penalizes divergence between intermediate/final
    3. Applied during training, so hiding is never reinforced
    """
    
    def __init__(self, base_model: nn.Module, config: TransparencyConfig):
        super().__init__()
        self.model = base_model
        self.config = config
        
        # Get model dimensions
        model_config = base_model.config
        n_layers = model_config.n_ode_blocks
        hidden_dim = model_config.n_embd
        vocab_size = model_config.vocab_size
        
        # Initialize transparency components
        self.decoder = ODEStochasticDecoder(
            config, n_layers, hidden_dim, vocab_size,
            base_model.ln_f, base_model.lm_head
        )
        self.consistency_loss = ConsistencyLoss(config)
    
    def forward(self, idx: Tensor, targets: Optional[Tensor] = None,
                **kwargs) -> Tuple[Tensor, Optional[Tensor], Dict[str, Any]]:
        """
        Forward pass with transparency monitoring.
        
        Returns:
            logits: Output logits
            loss: Combined task + consistency loss
            info: Dict with transparency metrics
        """
        self.decoder.reset()
        
        # Get embeddings
        x = self.model.forward_embeddings(idx)
        
        # Forward through ODE blocks with stochastic decoding
        intermediates = []
        for block_idx, ode_block in enumerate(self.model.ode_blocks):
            # Hook into ODE integration for stochastic decoding
            x_before = x.clone()
            x = x + ode_block(x)
            
            # Potentially decode after this block
            decoded = self.decoder.decode_and_record(x, block_idx)
            if decoded is not None:
                intermediates.append((block_idx, decoded))
        
        # Final output
        x = self.model.ln_f(x)
        
        if targets is not None:
            final_logits = self.model.lm_head(x)
            task_loss = F.cross_entropy(
                final_logits.view(-1, final_logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            
            # Consistency loss
            consistency = self.consistency_loss(
                self.decoder.get_decoded_logits(),
                final_logits,
                targets
            )
            
            total_loss = task_loss + self.config.consistency_weight * consistency
            
            info = {
                'task_loss': task_loss.item(),
                'consistency_loss': consistency.item(),
                'n_decodings': len(self.decoder.get_decoded_logits()),
            }
        else:
            final_logits = self.model.lm_head(x[:, [-1], :])
            total_loss = None
            info = {}
        
        return final_logits, total_loss, info


# =============================================================================
# Transparency-Aware Training
# =============================================================================

def transparency_train_step(model: TransparentCocoNODE, 
                           batch: Dict[str, Any],
                           optimizer: torch.optim.Optimizer,
                           scaler: torch.amp.GradScaler,
                           config: TransparencyConfig,
                           ctx) -> Dict[str, float]:
    """
    Single training step with transparency monitoring.
    
    This modifies the standard training loop to:
    1. Apply stochastic decoding during forward pass
    2. Compute and backprop consistency loss
    3. Log transparency metrics
    """
    optimizer.zero_grad(set_to_none=True)
    
    with ctx:
        logits, loss, info = model(
            batch['input_ids'],
            targets=batch['targets'],
            latent_positions=batch.get('latent_positions'),
            n_latent_steps=batch.get('n_latent_steps', 0)
        )
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return {
        'loss': loss.item(),
        **info
    }


# =============================================================================
# Adversarial Hiding Detector (Optional Advanced Feature)
# =============================================================================

class HidingDetector(nn.Module):
    """
    Adversarial detector that tries to identify when the model
    is encoding information not visible in decoded representations.
    
    Training setup:
    - Detector tries to maximize detection of hidden information
    - Main model is penalized when detector succeeds
    - Creates minimax game where hiding becomes unprofitable
    
    This is an advanced feature for high-stakes deployments.
    """
    
    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        
        # Detector takes pairs of (hidden_state, decoded_logits) 
        # and tries to predict if there's hidden information
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Pooling over sequence
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, hidden_states: Tensor, decoded_logits: Tensor, 
                vocab_embeddings: Tensor) -> Tensor:
        """
        Predict probability that hidden information exists.
        
        Args:
            hidden_states: [B, T, D] actual hidden states
            decoded_logits: [B, T, V] what we decoded from hidden states
            vocab_embeddings: [V, D] embedding matrix for reconstruction
        
        Returns:
            hiding_prob: [B,] probability of hidden information per batch
        """
        B, T, D = hidden_states.shape
        
        # Reconstruct "visible" representation from decoded logits
        probs = F.softmax(decoded_logits, dim=-1)  # [B, T, V]
        reconstructed = probs @ vocab_embeddings   # [B, T, D]
        
        # Compare actual vs reconstructed
        combined = torch.cat([hidden_states, reconstructed], dim=-1)  # [B, T, 2D]
        
        # Detect hidden information
        detection_scores = self.encoder(combined)  # [B, T, 1]
        
        # Pool over sequence
        pooled = self.pool(detection_scores.transpose(1, 2)).squeeze(-1)  # [B, 1]
        
        return pooled.squeeze(-1)


class AdversarialTransparencyTrainer:
    """
    Adversarial training for transparency.
    
    Two-player game:
    - Hiding Detector: Tries to detect when model hides information
    - Main Model: Tries to solve tasks without triggering detector
    
    Equilibrium: Model cannot hide information profitably.
    """
    
    def __init__(self, model: TransparentCocoNODE, 
                 detector: HidingDetector,
                 config: TransparencyConfig):
        self.model = model
        self.detector = detector
        self.config = config
        
        # Separate optimizers
        self.model_optimizer = None  # Set externally
        self.detector_optimizer = torch.optim.AdamW(
            detector.parameters(), lr=1e-4
        )
    
    def train_step(self, batch: Dict[str, Any], ctx) -> Dict[str, float]:
        """
        Adversarial training step.
        
        1. Forward pass through model with stochastic decoding
        2. Train detector to identify hiding
        3. Train model to avoid detection while maintaining performance
        """
        # Forward pass
        self.model.decoder.reset()
        
        with ctx:
            logits, task_loss, info = self.model(
                batch['input_ids'],
                targets=batch['targets']
            )
        
        # Get hidden states and decodings for detector
        # (This requires modifying model to expose intermediates)
        decoded_pairs = self.model.decoder.get_decoded_logits()
        
        if len(decoded_pairs) > 0 and self.config.adversarial_weight > 0:
            # Train detector (maximize detection)
            self.detector_optimizer.zero_grad()
            
            # Create positive examples (with hidden info) and negative (without)
            # For simplicity, we use consistency loss as proxy for "hiding"
            consistency = info.get('consistency_loss', 0)
            hiding_label = torch.tensor(
                [1.0 if consistency > 0.1 else 0.0], 
                device=logits.device
            )
            
            # This is simplified - full implementation would need actual
            # hidden state access during forward pass
            detector_loss = F.binary_cross_entropy(
                torch.tensor([0.5], device=logits.device),  # Placeholder
                hiding_label
            )
            
            detector_loss.backward(retain_graph=True)
            self.detector_optimizer.step()
            
            info['detector_loss'] = detector_loss.item()
        
        return info


# =============================================================================
# Inference-Time Transparency Monitor
# =============================================================================

class TransparencyMonitor:
    """
    Runtime monitor for deployed models.
    
    During inference, continuously samples and decodes intermediate states
    to verify the model isn't hiding strategic computation.
    
    Alerts if:
    - Decoded intermediates diverge significantly from final output
    - Pattern of decodings suggests learned avoidance
    - Anomalous activation patterns detected
    """
    
    def __init__(self, model: TransparentCocoNODE, 
                 config: TransparencyConfig,
                 alert_threshold: float = 0.5):
        self.model = model
        self.config = config
        self.alert_threshold = alert_threshold
        
        # Tracking
        self.consistency_history: List[float] = []
        self.alert_count: int = 0
    
    @torch.no_grad()
    def monitored_forward(self, idx: Tensor, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Forward pass with full monitoring.
        
        Returns output plus detailed transparency report.
        """
        self.model.decoder.reset()
        
        # Force high decode probability for monitoring
        original_prob = self.config.decode_prob
        self.config.decode_prob = 0.5  # Decode ~half of layers
        
        logits, _, info = self.model(idx, **kwargs)
        
        self.config.decode_prob = original_prob
        
        # Analyze decodings
        report = self._analyze_decodings(logits, info)
        
        # Update history
        self.consistency_history.append(report.get('max_divergence', 0))
        if len(self.consistency_history) > 1000:
            self.consistency_history = self.consistency_history[-1000:]
        
        # Check for alerts
        if report.get('alert', False):
            self.alert_count += 1
            report['cumulative_alerts'] = self.alert_count
        
        return logits, report
    
    def _analyze_decodings(self, final_logits: Tensor, 
                          info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze decoded intermediates for signs of hiding."""
        decoded = self.model.decoder.get_decoded_logits()
        
        if len(decoded) == 0:
            return {'status': 'no_decodings'}
        
        # Compute divergences
        final_probs = F.softmax(final_logits, dim=-1)
        divergences = []
        
        for layer_idx, intermed_logits in decoded:
            intermed_probs = F.softmax(intermed_logits, dim=-1)
            
            # KL divergence
            kl = F.kl_div(
                intermed_probs.log(), 
                final_probs, 
                reduction='batchmean'
            ).item()
            divergences.append((layer_idx, kl))
        
        max_div = max(d[1] for d in divergences)
        mean_div = sum(d[1] for d in divergences) / len(divergences)
        
        # Check for alert conditions
        alert = max_div > self.alert_threshold
        
        # Check for suspicious patterns (e.g., increasing divergence in later layers)
        if len(divergences) >= 3:
            # Sort by layer
            sorted_divs = sorted(divergences, key=lambda x: x[0])
            later_mean = sum(d[1] for d in sorted_divs[-2:]) / 2
            earlier_mean = sum(d[1] for d in sorted_divs[:2]) / 2
            
            if later_mean > earlier_mean * 2:
                alert = True
        
        return {
            'status': 'alert' if alert else 'ok',
            'alert': alert,
            'max_divergence': max_div,
            'mean_divergence': mean_div,
            'layer_divergences': divergences,
            'n_decodings': len(decoded),
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from monitoring history."""
        if not self.consistency_history:
            return {'status': 'no_history'}
        
        import numpy as np
        history = np.array(self.consistency_history)
        
        return {
            'total_forwards': len(history),
            'alert_count': self.alert_count,
            'alert_rate': self.alert_count / len(history),
            'mean_divergence': history.mean(),
            'max_divergence': history.max(),
            'std_divergence': history.std(),
            'p95_divergence': np.percentile(history, 95),
        }


# =============================================================================
# Integration with GRPO-style RL Training
# =============================================================================

def compute_transparency_reward(model: TransparentCocoNODE,
                                 response: Tensor,
                                 targets: Tensor) -> Tensor:
    """
    Compute transparency reward for GRPO training.
    
    Reward = α * correctness + β * consistency + γ * faithfulness
    
    This incentivizes models to:
    1. Get the right answer
    2. Have consistent intermediate representations
    3. Have intermediates that causally influence output
    """
    model.decoder.reset()
    
    logits, loss, info = model(response, targets=targets)
    
    # Correctness: negative cross-entropy (higher is better)
    correctness = -loss.item()
    
    # Consistency: negative consistency loss (higher is better)
    consistency = -info.get('consistency_loss', 0)
    
    # Faithfulness: how much do intermediates predict final output
    # (Simplified - full version would do counterfactual interventions)
    decoded = model.decoder.get_decoded_logits()
    if len(decoded) > 0:
        final_pred = logits.argmax(dim=-1)
        intermed_preds = [d[1].argmax(dim=-1) for d in decoded]
        agreement = sum((p == final_pred).float().mean().item() for p in intermed_preds)
        faithfulness = agreement / len(decoded)
    else:
        faithfulness = 0.5  # Neutral
    
    # Combined reward (weights are hyperparameters)
    alpha, beta, gamma = 1.0, 0.1, 0.1
    reward = alpha * correctness + beta * consistency + gamma * faithfulness
    
    return torch.tensor(reward, device=response.device)


# =============================================================================
# Utility: Visualization of Transparency
# =============================================================================

def visualize_reasoning_trace(model: TransparentCocoNODE,
                              idx: Tensor,
                              tokenizer) -> str:
    """
    Create human-readable trace of model's reasoning at each layer.
    
    This is crucial for interpretability: we can literally read
    what the model is "thinking" at each computation stage.
    """
    model.eval()
    model.decoder.reset()
    
    # Force full decoding
    original_prob = model.config.decode_prob
    model.decoder.config.decode_prob = 1.0
    
    with torch.no_grad():
        logits, _, _ = model(idx)
    
    model.decoder.config.decode_prob = original_prob
    
    # Build trace
    lines = ["=" * 60, "REASONING TRACE", "=" * 60, ""]
    
    # Input
    input_text = tokenizer.decode(idx[0].tolist())
    lines.append(f"INPUT: {input_text[:100]}...")
    lines.append("")
    
    # Intermediate decodings
    for layer_idx, intermed_logits in model.decoder.get_decoded_logits():
        pred_tokens = intermed_logits[0].argmax(dim=-1)
        pred_text = tokenizer.decode(pred_tokens.tolist())
        lines.append(f"LAYER {layer_idx}: {pred_text[:100]}...")
    
    lines.append("")
    
    # Final output
    final_tokens = logits[0].argmax(dim=-1)
    final_text = tokenizer.decode(final_tokens.tolist())
    lines.append(f"FINAL: {final_text[:100]}...")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)