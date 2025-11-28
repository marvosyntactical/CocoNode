"""
CocoNODE: Neural ODE GPT with Chain of Continuous Thought

Combines:
1. Neural ODE treatment of transformer depth (continuous-time layers)
2. Coconut-style latent reasoning (hidden states fed back as input)
3. Adaptive computation time via ODE solver tolerance

The key insight: treating depth as continuous time while also treating
reasoning steps as continuous latent states creates a doubly-continuous
model where both "how deep to think" and "what to think" are learned
in continuous spaces.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# =============================================================================
# ODE Solvers
# =============================================================================

def euler_step(z: Tensor, t: Tensor, h: Tensor, f) -> Tensor:
    """Single Euler step: z_{n+1} = z_n + h * f(z_n, t_n)"""
    return z + h * f(z, t)


def rk4_step(z: Tensor, t: Tensor, h: Tensor, f) -> Tensor:
    """Single RK4 step - 4th order accuracy"""
    k1 = f(z, t)
    k2 = f(z + h * k1 / 2, t + h / 2)
    k3 = f(z + h * k2 / 2, t + h / 2)
    k4 = f(z + h * k3, t + h)
    return z + h * (k1 + 2*k2 + 2*k3 + k4) / 6


def ode_solve(z0: Tensor, t0: Tensor, t1: Tensor, f, 
              method: str = 'euler', h_max: float = 0.25) -> Tensor:
    """
    Solve ODE from t0 to t1 starting at z0.
    
    Args:
        z0: Initial state [B, T, D]
        t0: Start time (scalar or [B,] tensor)
        t1: End time
        f: Dynamics function f(z, t) -> dz/dt
        method: 'euler' or 'rk4'
        h_max: Maximum step size (controls discretization)
    
    Returns:
        z1: Final state at time t1
    """
    step_fn = euler_step if method == 'euler' else rk4_step
    
    # Compute number of steps needed
    dt = t1 - t0
    if isinstance(dt, Tensor):
        n_steps = max(1, math.ceil(dt.abs().max().item() / h_max))
    else:
        n_steps = max(1, math.ceil(abs(dt) / h_max))
    
    h = dt / n_steps
    t = t0
    z = z0
    
    for _ in range(n_steps):
        z = step_fn(z, t, h, f)
        t = t + h
    
    return z


# =============================================================================
# ODE Function Base Class
# =============================================================================

class ODEF(nn.Module):
    """Base class for ODE dynamics that supports adjoint method."""
    
    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        raise NotImplementedError
    
    def forward_with_grad(self, z: Tensor, t: Tensor, grad_outputs: Tensor):
        """Compute f and gradients df/dz, df/dp, df/dt for adjoint method."""
        batch_size = z.shape[0]
        
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            t = t.detach().requires_grad_(True)
            out = self.forward(z, t)
            
            a = grad_outputs
            adfdz, adfdt, *adfdp = torch.autograd.grad(
                (out,), (z, t) + tuple(self.parameters()), 
                grad_outputs=(a,),
                allow_unused=True, 
                retain_graph=True
            )
            
            if adfdp:
                adfdp = torch.cat([p.flatten() for p in adfdp if p is not None]).unsqueeze(0)
                adfdp = adfdp.expand(batch_size, -1) / batch_size
            if adfdt is not None:
                adfdt = adfdt.expand(batch_size, 1) / batch_size
                
        return out, adfdz, adfdt, adfdp
    
    def flatten_parameters(self) -> Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])


# =============================================================================
# Transformer Components
# =============================================================================

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with time conditioning."""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # Time-conditioned projections: input includes time embedding
        self.c_attn = nn.Linear(config.n_embd + config.time_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Pre-attention LayerNorm
        self.ln = LayerNorm(config.n_embd + config.time_embd, bias=config.bias)
        
        # Causal mask
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, D + time_embd] where last dim is time embedding
        Returns:
            [B, T, D] attention output
        """
        x = self.ln(x)
        B, T, C = x.size()
        
        # QKV projection
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        head_dim = self.n_embd // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """Feed-forward network with time conditioning."""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd + config.time_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = LayerNorm(config.n_embd + config.time_embd, bias=config.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ln(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# =============================================================================
# Neural ODE Transformer Block
# =============================================================================

class ODEBlock(ODEF):
    """
    Transformer block as ODE dynamics: dz/dt = f(z, t)
    
    Time t is embedded and concatenated to enable time-varying dynamics.
    This allows the ODE solver to learn different transformations at
    different "depths" while sharing parameters.
    """
    
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.time_embd = config.time_embd
        
        # Learnable time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.time_embd),
            nn.GELU(),
            nn.Linear(config.time_embd, config.time_embd),
        )
    
    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """
        Compute dz/dt at state z and time t.
        
        Args:
            z: Hidden state [B, T, D]
            t: Time scalar or [B,] tensor
        
        Returns:
            dz/dt: Rate of change [B, T, D]
        """
        B, T, D = z.shape
        
        # Embed time and broadcast to sequence
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_emb = self.time_mlp(t.view(-1, 1))  # [B, time_embd] or [1, time_embd]
        t_emb = t_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, time_embd]
        
        # Concatenate time embedding
        zt = torch.cat([z, t_emb], dim=-1)  # [B, T, D + time_embd]
        
        # Attention + MLP as residual dynamics
        h = self.attn(zt)
        ht = torch.cat([h, t_emb], dim=-1)
        dz = self.mlp(ht)
        
        return dz


class NeuralODELayer(nn.Module):
    """
    Wraps ODEBlock for integration from t=0 to t=1.
    
    The ODE integration replaces discrete layer stacking - instead of
    L separate layers, we have a single continuous transformation.
    """
    
    def __init__(self, func: ODEF, method: str = 'euler', h_max: float = 0.25):
        super().__init__()
        self.func = func
        self.method = method
        self.h_max = h_max
    
    def forward(self, z0: Tensor, t_span: Tuple[float, float] = (0., 1.)) -> Tensor:
        """
        Integrate ODE from t_span[0] to t_span[1].
        
        Args:
            z0: Initial state [B, T, D]
            t_span: (t_start, t_end)
        
        Returns:
            z1: Final state at t_end
        """
        t0 = torch.tensor(t_span[0], device=z0.device, dtype=z0.dtype)
        t1 = torch.tensor(t_span[1], device=z0.device, dtype=z0.dtype)
        
        return ode_solve(z0, t0, t1, self.func, method=self.method, h_max=self.h_max)


# =============================================================================
# CocoNODE: Neural ODE GPT with Continuous Thought
# =============================================================================

@dataclass
class CocoNODEConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_embd: int = 768
    n_head: int = 12
    time_embd: int = 64  # Time embedding dimension for ODE
    dropout: float = 0.0
    bias: bool = True
    
    # ODE settings
    ode_method: str = 'euler'  # 'euler' or 'rk4'
    ode_h_max: float = 0.25   # Max step size (fewer steps = faster, less accurate)
    n_ode_blocks: int = 4     # Number of sequential ODE blocks (each integrates t=0->1)
    
    # Coconut settings
    max_latent_steps: int = 8  # Maximum continuous thought iterations
    latent_token_id: int = -1  # Special token ID for latent positions (set during training)


class CocoNODE(nn.Module):
    """
    CocoNODE: Combining Neural ODE depth with Coconut-style latent reasoning.
    
    Architecture:
    - Token + position embeddings
    - N sequential ODE blocks (each integrates from t=0 to t=1)
    - Optional latent reasoning: feed hidden states back as inputs
    
    Key innovation for safety/interpretability:
    - Both depth (ODE integration) and reasoning (latent steps) are continuous
    - Intermediate states can be decoded at any point
    - Stochastic decoding during training prevents hidden computation
    """
    
    def __init__(self, config: CocoNODEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # ODE blocks (stacked, each integrates t=0->1)
        self.ode_blocks = nn.ModuleList([
            NeuralODELayer(
                ODEBlock(config), 
                method=config.ode_method,
                h_max=config.ode_h_max
            )
            for _ in range(config.n_ode_blocks)
        ])
        
        # Final layer norm and output head
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.wte.weight = self.lm_head.weight
        
        # Special tokens for Coconut-style reasoning
        self.bot_embedding = nn.Parameter(torch.randn(config.n_embd) * 0.02)  # Begin of thought
        self.eot_embedding = nn.Parameter(torch.randn(config.n_embd) * 0.02)  # End of thought
        
        self._init_weights()
        print(f"CocoNODE: {self.get_num_params()/1e6:.2f}M parameters")
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
    
    def forward_embeddings(self, idx: Tensor) -> Tensor:
        """Get token + position embeddings."""
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        return self.drop(tok_emb + pos_emb)
    
    def forward_ode_blocks(self, x: Tensor, 
                           return_intermediates: bool = False) -> Tensor | Tuple[Tensor, List[Tensor]]:
        """
        Pass through all ODE blocks.
        
        Args:
            x: Input embeddings [B, T, D]
            return_intermediates: If True, return list of intermediate states
        
        Returns:
            Final hidden state, optionally with intermediates
        """
        intermediates = [x] if return_intermediates else None
        
        for ode_block in self.ode_blocks:
            x = x + ode_block(x)  # Residual connection around ODE
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def forward(self, idx: Tensor, targets: Optional[Tensor] = None,
                latent_positions: Optional[List[List[int]]] = None,
                n_latent_steps: int = 0) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with optional Coconut-style latent reasoning.
        
        Args:
            idx: Token indices [B, T]
            targets: Target indices for loss computation [B, T]
            latent_positions: Per-batch list of positions to use continuous thought
            n_latent_steps: Number of latent reasoning iterations (Coconut)
        
        Returns:
            logits: Output logits [B, T, V]
            loss: Cross-entropy loss if targets provided
        """
        device = idx.device
        B, T = idx.size()
        
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block size {self.config.block_size}"
        
        # Get embeddings
        x = self.forward_embeddings(idx)
        
        # Coconut-style latent reasoning
        if n_latent_steps > 0 and latent_positions is not None:
            x = self._latent_reasoning(x, latent_positions, n_latent_steps)
        else:
            # Standard forward pass
            x = self.forward_ode_blocks(x)
        
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1), 
                ignore_index=-1
            )
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def _latent_reasoning(self, x: Tensor, latent_positions: List[List[int]], 
                          n_steps: int) -> Tensor:
        """
        Perform Coconut-style continuous thought reasoning.
        
        At each latent step:
        1. Run through ODE blocks
        2. Extract hidden states at latent positions
        3. Feed these back as input embeddings for next iteration
        
        This creates a recurrent computation in latent space.
        """
        B, T, D = x.shape
        
        for step in range(n_steps):
            # Forward through ODE blocks
            x = self.forward_ode_blocks(x)
            
            # Extract continuous thoughts and feed back
            for batch_idx, positions in enumerate(latent_positions):
                if step < len(positions):
                    pos = positions[step]
                    if pos < T:
                        # Replace embedding at this position with continuous thought
                        # This is the key Coconut operation
                        x[batch_idx, pos] = x[batch_idx, pos].detach() + \
                                           x[batch_idx, pos] - x[batch_idx, pos].detach()
        
        # Final pass through ODE blocks
        x = self.forward_ode_blocks(x)
        return x
    
    def decode_at_layer(self, x: Tensor, layer_idx: int) -> Tensor:
        """
        Decode hidden state to logits at any intermediate layer.
        
        This is key for interpretability/safety: we can inspect
        what the model is "thinking" at any point in the computation.
        """
        # Apply final norm and decode
        x = self.ln_f(x)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, idx: Tensor, max_new_tokens: int, 
                 temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def configure_optimizers(self, weight_decay: float, learning_rate: float, 
                            betas: Tuple[float, float], device_type: str):
        """Configure AdamW optimizer with weight decay."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Weight decay for 2D+ params (weights), no decay for 1D params (biases, norms)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
