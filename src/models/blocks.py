from functools import partial
import math
from typing import List, Optional

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# Settings for GroupNorm and Attention

GN_GROUP_SIZE = 32
GN_EPS = 1e-5
ATTN_HEAD_DIM = 8

# Convs

Conv1x1 = partial(nn.Conv2d, kernel_size=1, stride=1, padding=0)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, stride=1, padding=1)

# GroupNorm and conditional GroupNorm


class GroupNorm(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=GN_EPS)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = max(1, in_channels // GN_GROUP_SIZE)
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """ 
        c: in_channels
        t*e: cond_channels
        cond shape: (b, t*e)
        x shape: (b, c, h, w)
        """
        assert x.size(1) == self.in_channels, f"Expected {self.in_channels} channels, got {x.size(1)}"
        x = F.group_norm(x, self.num_groups, eps=GN_EPS)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


# Self Attention for 2D images 
# language embedding (dim,) sliced into multiple heads, temporal attention between tokens 
# image channels (channel, h * w) sliced into multiple heads, spatial attention between patches (still 2D matrix attention by flattening image)

class SelfAttention2d(nn.Module): # 2D attention? Cost of this must be quadratic with # frames ... (!)
    def __init__(self, in_channels: int, head_dim: int = ATTN_HEAD_DIM) -> None:
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = Conv1x1(in_channels, in_channels * 3)
        self.out_proj = Conv1x1(in_channels, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape # c = in_channels
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        qkv = qkv.view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3).contiguous() 
        q, k, v = [x for x in qkv.chunk(3, dim=1)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(2, 3).reshape(n, c, h, w)
        return x + self.out_proj(y)


# Embedding of the noise level


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: Tensor) -> Tensor:
        """ 
        Input shape: (batch_size,)
        Output shape: (batch_size, cond_channels)
        """
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


# [Down|Up]sampling | Scale [up|down] spatial dimension by 2


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        nn.init.orthogonal_(self.conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# Small Residual block


class SmallResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.f = nn.Sequential(GroupNorm(in_channels), nn.SiLU(inplace=True), Conv3x3(in_channels, out_channels))
        self.skip_projection = nn.Identity() if in_channels == out_channels else Conv1x1(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.skip_projection(x) + self.f(x)


# Residual block (conditioning with AdaGroupNorm, no [down|up]sampling, optional self-attention)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main path with correct channel counts after input projection
        self.norm1 = AdaGroupNorm(in_channels, cond_channels)  # Expect channels[0] not 1
        self.conv1 = Conv3x3(in_channels, out_channels)
        self.norm2 = AdaGroupNorm(out_channels, cond_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)
        
        # Skip connection with projection if needed
        self.skip_proj = Conv1x1(in_channels, out_channels) if in_channels != out_channels else None
        
        # Optional attention
        self.attn = SelfAttention2d(out_channels) if attn else None
        
        # Initialize final conv to zero
        nn.init.zeros_(self.conv2.weight)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        identity = x
        
        # Main path
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        
        # Skip connection with optional projection
        if self.skip_proj is not None:
            identity = self.skip_proj(identity)
        x = x + identity
        
        # Optional attention
        if self.attn is not None:
            x = self.attn(x)
            
        return x


# Sequence of residual blocks (in_channels -> mid_channels -> ... -> mid_channels -> out_channels)


class ResBlocks(nn.Module):
    def __init__(
        self,
        list_in_channels: List[int],
        list_out_channels: List[int],
        cond_channels: int,
        attn: bool,
        skip_connection: bool = True,
    ) -> None:
        super().__init__()
        assert len(list_in_channels) == len(list_out_channels)
        self.in_channels = list_in_channels[0]
        self.skip_connection = skip_connection
        
        
        # Ensure each ResBlock gets the correct input channels
        self.resblocks = nn.ModuleList()
        prev_out_channels = list_in_channels[0]
        for i, (in_ch, out_ch) in enumerate(zip(list_in_channels, list_out_channels)):
            # For blocks after the first one, input channels should match previous block's output
            actual_in_channels = prev_out_channels if i > 0 else in_ch
            self.resblocks.append(
                ResBlock(
                    in_channels=actual_in_channels,
                    out_channels=out_ch,
                    cond_channels=cond_channels,
                    attn=attn
                )
            )
            prev_out_channels = out_ch

    def forward(self, x: Tensor, cond: Tensor, to_cat: Optional[List[Tensor]] = None) -> Tensor:
        outputs = []
        for i, resblock in enumerate(self.resblocks):
            # Handle skip connections if provided
            if self.skip_connection and to_cat is not None and i > 0:
                skip = to_cat[i-1]
                # Project skip connection to match current channels if needed
                if skip.size(1) != x.size(1):

                    skip_proj = Conv1x1(skip.size(1), x.size(1)).to(x.device)
                    skip = skip_proj(skip)
                x = x + skip  # Add skip connection instead of concatenating
                
            # Process through ResBlock
            x = resblock(x, cond)
            outputs.append(x)
            
        return x, outputs


# UNet || To be interpreted
# - len(depths) controls scaling of spatial resolutions (upsample / downsample)
# - "n" controls computation power for each upsample | downsample block 
# - input, as well as "n" intermediate outputs from downsample block are passed into each upsample block as skip connections
# - overall, UNet shrinks and expands spatial resolution, while enabling skip connections for retaining information
# - special ResBlock contains attention and action conditioning operation

class UNet(nn.Module):
    def __init__(self, cond_channels: int, depths: List[int], channels: List[int], attn_depths: List[int], in_channels: int = 1) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)
        self._num_down = len(channels) - 1
        self.in_channels = in_channels
        
        # Always project input to initial working channels
        self.input_proj = Conv1x1(in_channels, channels[0])


        d_blocks, u_blocks = [], [] # [down-sampling | up-sampling] each has 'depths' number of blocks
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            
            # For downsampling blocks with proper channel growth
            if i == 0:
                # First block after input projection: channels[0] -> channels[0]
                d_in_channels = [channels[0]] * n
                d_out_channels = [channels[0]] * n
            else:
                # Subsequent blocks: channels[i-1] -> channels[i]
                d_in_channels = [channels[i-1]] * n  # All blocks get same input channels
                d_out_channels = [channels[i]] * n   # All blocks output same channels
            
            
            d_blocks.append(
                ResBlocks(
                    list_in_channels=d_in_channels,
                    list_out_channels=d_out_channels,
                    skip_connection=True,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            )
            
            # For upsampling blocks, use channel sizes directly from channels list
            u_in_channels = [channels[i]] * n + [channels[max(0, i-1)]]
            u_out_channels = [channels[i]] * n + [channels[max(0, i-1)]]
            
            u_blocks.append(
                ResBlocks(
                    list_in_channels=u_in_channels,
                    list_out_channels=u_out_channels,
                    cond_channels=cond_channels,
                    attn=attn_depths[i],
                )
            ) # Upsample block take residual from down-sampling block, extra channels in input required
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(reversed(u_blocks))

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=cond_channels,
            attn=True,
        )

        # Create downsampling layers with proper channel counts
        downsamples = [nn.Identity()]  # First layer doesn't downsample
        for i in range(len(channels)-1):
            # Downsample should use input channel count
            in_ch = channels[i]
            downsamples.append(Downsample(in_ch))
            
        # Create upsampling layers with proper channel counts
        upsamples = [nn.Identity()]  # First layer doesn't upsample
        for i in reversed(range(len(channels)-1)):
            # Upsample should use output channel count
            out_ch = channels[i]
            upsamples.append(Upsample(out_ch))
            
        self.downsamples = nn.ModuleList(downsamples)
        self.upsamples = nn.ModuleList(upsamples)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # Project input to initial working channels
        x = self.input_proj(x)
        
        # Handle padding for proper downsampling
        *_, h, w = x.size()
        n = self._num_down
        padding_h = math.ceil(h / 2 ** n) * 2 ** n - h
        padding_w = math.ceil(w / 2 ** n) * 2 ** n - w
        x = F.pad(x, (0, padding_w, 0, padding_h))

        
        # Store original x for skip connections
        x_orig = x

        # Downsampling path
        d_outputs = []
        x_current = x_orig  # Start with original input
        for i, (block, down) in enumerate(zip(self.d_blocks, self.downsamples)):
            # Apply downsampling first
            x_down = down(x_current)

            
            # Process through residual blocks
            x_processed, block_outputs = block(x_down, cond)

            
            # Store outputs and update current tensor
            d_outputs.append((x_down, *block_outputs))
            x_current = x_processed

        # Middle blocks
        x_mid, _ = self.mid_blocks(x_current, cond)
        
        # Upsampling path
        u_outputs = []
        x_current = x_mid  # Start upsampling from mid block output
        for i, (block, up, skip) in enumerate(zip(self.u_blocks, self.upsamples, reversed(d_outputs))):
            # Apply upsampling
            x_up = up(x_current)
            try:
                # Process through residual blocks with skip connections
                x_processed, block_outputs = block(x_up, cond, skip[::-1])
                # Store outputs and update current tensor
                u_outputs.append((x_up, *block_outputs))
                x_current = x_processed
            except Exception as e:

                raise e

        # Remove padding and return final output
        x_final = x_current[..., :h, :w]

        return x_final, d_outputs, u_outputs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    