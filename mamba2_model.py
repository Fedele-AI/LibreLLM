"""
Mamba-2 Implementation in PyTorch
Pure PyTorch implementation that works on CPU/GPU without CUDA kernels.
Based on the Agora-Lab-AI implementation.
"""
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor


class DeviceType(Enum):
    """Supported device types for model execution."""
    CPU = auto()
    GPU = auto()


@dataclass
class Mamba2Config:
    """Configuration for Mamba-2 model.
    
    Args:
        d_model: Model dimension
        depth: Number of Mamba blocks
        d_state: State dimension for SSM
        d_conv: Convolution kernel size
        expand_factor: Expansion factor for inner dimension
        vocab_size: Vocabulary size for embeddings
        device_type: Type of device to run on
        dtype: Data type for model parameters
        max_seq_len: Maximum sequence length (8k tokens)
    """
    d_model: int
    depth: int
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2
    vocab_size: int = 50257  # GPT-2 vocab size
    device_type: DeviceType = DeviceType.CPU
    dtype: torch.dtype = torch.float32
    max_seq_len: int = 8192  # 8k context length

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be positive, got {self.d_state}")
        if self.d_conv <= 0:
            raise ValueError(f"d_conv must be positive, got {self.d_conv}")
        if self.expand_factor <= 0:
            raise ValueError(f"expand_factor must be positive, got {self.expand_factor}")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class Mamba2Block(nn.Module):
    """Single Mamba-2 block with state-space model."""
    
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        self.d_inner = config.d_model * config.expand_factor
        
        # Layer norm
        self.norm = RMSNorm(config.d_model)
        
        # Input projection
        self.in_proj = nn.Linear(config.d_model, self.d_inner * 2, bias=False)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.d_conv,
            groups=self.d_inner,
            padding=config.d_conv - 1,
            bias=True
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, config.d_state, bias=False)
        self.dt_proj = nn.Linear(config.d_state, self.d_inner, bias=True)
        
        # State-space matrices (simplified)
        self.A = nn.Parameter(torch.randn(self.d_inner, config.d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, config.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, config.d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, config.d_model, bias=False)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through Mamba block."""
        batch, seq_len, d_model = x.shape
        
        # Normalize
        x = self.norm(x)
        
        # Input projection - split into two paths
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # Convolution (needs transpose for Conv1d)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = torch.nn.functional.silu(x_conv)
        
        # Simplified SSM computation
        # Project to state space: [batch, seq_len, d_inner] -> [batch, seq_len, d_state]
        x_state = self.x_proj(x_conv)  # [batch, seq_len, d_state]
        
        # Compute timesteps
        delta = torch.nn.functional.softplus(self.dt_proj(x_state))  # [batch, seq_len, d_inner]
        
        # Simplified state-space: replace complex selective scan with linear projection
        # This is a simplified version that maintains dimensionality
        # [batch, seq_len, d_state] -> [batch, seq_len, d_inner]
        
        # Linear combination with state-space parameters
        # B projection: [batch, seq_len, d_state] @ [d_state, d_inner] = [batch, seq_len, d_inner]
        y_state = x_state @ self.B.t()  # [batch, seq_len, d_inner]
        
        # Modulate by delta (timestep)
        y_state = y_state * delta
        
        # Add direct path (skip connection in state space)
        y = y_state + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        # Gate with z (gating mechanism)
        y = y * torch.nn.functional.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class Mamba2LM(nn.Module):
    """Complete Mamba-2 Language Model."""
    
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            Mamba2Block(config) for _ in range(config.depth)
        ])
        
        # Final norm
        self.norm_f = RMSNorm(config.d_model)
        
        # Language model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: Tensor, labels: Optional[Tensor] = None):
        """Forward pass.
        
        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target token IDs for loss computation [batch, seq_len]
            
        Returns:
            logits or (loss, logits)
        """
        # Embed tokens
        x = self.embeddings(input_ids)
        
        # Pass through Mamba blocks with residual connections
        for block in self.blocks:
            x = x + block(x)
        
        # Final normalization
        x = self.norm_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss, logits
        
        return logits
    
    def generate(self, input_ids: Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: int = 50):
        """Generate tokens autoregressively.
        
        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated token IDs
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate if needed
                idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]
                
                # Forward pass
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append
                input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids


def create_mamba2_model(
    d_model: int = 256,
    depth: int = 6,
    vocab_size: int = 50257,
    max_seq_len: int = 8192,
    device: str = "cpu"
) -> Mamba2LM:
    """Create a Mamba-2 language model.
    
    Args:
        d_model: Model dimension
        depth: Number of layers
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length (8k tokens)
        device: Device to place model on
        
    Returns:
        Mamba2LM model
    """
    config = Mamba2Config(
        d_model=d_model,
        depth=depth,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        device_type=DeviceType.GPU if device == "cuda" else DeviceType.CPU
    )
    
    model = Mamba2LM(config)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created Mamba-2 model with {n_params:,} parameters")
    print(f"Max sequence length: {max_seq_len} tokens")
    
    return model
