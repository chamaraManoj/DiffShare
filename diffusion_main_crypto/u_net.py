'''
Torch architecture with U-net
'''

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

from labml_helpers.module import Module
import numpy as np

# the  weight matrix which weigh the previous chunk values
def get_weight_matrix(num_of_rows, num_of_cols, device):
    x = np.arange(1, num_of_rows + 1)
    x = 1 / np.power(x, 2)

    x = x.reshape([-1, 1])
    y = np.tile(x, [1, num_of_cols])

    y = torch.from_numpy(y).float()
    y = y.to(device)
    return y

# activation function used in the models
class Swish(Module):
    """
    ### Swish activation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)

# time embeddiung generator which provided temporal information during the model training.
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    # just pass 256 or the required number of channel according to the convolution
    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class ResidualBlock(Module):
    """
    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups
        * `dropout` is the dropout rate
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # convolution previous chunk
        self.norm1_prev = nn.GroupNorm(n_groups, in_channels)
        self.act1_prev = Swish()
        self.conv1_prev = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            self.change_channels_x_prev = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            self.change_channels_x_prev.weight.data.fill_(0)  # assign the weights and other values
            self.change_channels_x_prev.bias.data.fill_(0)
        else:
            self.shortcut = nn.Identity()
            self.change_channels_x_prev = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        # Linear layer for relative position embeddings
        self.r_emb = nn.Linear(time_channels, out_channels)
        self.r_act = Swish()

        # Linear layer for relative position embeddings
        self.minmax_emb = nn.Linear(time_channels, out_channels)
        self.minmax_act = Swish()

        self.dropout = nn.Dropout(dropout)

        # initialize the weights and biases to 0
        self.norm1_prev.weight.data.fill_(0)
        self.norm1_prev.bias.data.fill_(0)
        self.conv1_prev.weight.data.fill_(0)
        self.conv1_prev.bias.data.fill_(0)

    # diffusion model denoising starts here
    def forward(self,
                x: torch.Tensor,
                x_prev: torch.Tensor,
                x_minmax: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer for the current chunk
        h = self.conv1(self.act1(self.norm1(x)))

        # First convolution layer for the previous chunk
        h += self.conv1_prev(self.act1_prev(self.norm1_prev(x_prev)))

        # Add time embeddings
        h += self.time_emb(self.time_act(t))[:, :, None, None]

        # Add relative chunk position embedding
        h += self.r_emb(self.r_act(r))[:, :, None, None]

        # Add relative min-max value embedding
        h += self.minmax_emb(self.minmax_act(x_minmax))[:, :, None, None]

        # Second convolution layer
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(Module):
    """
    ### Attention block used during the denoising process which helps learning the sequential patterns
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self,
                x: torch.Tensor,
                r: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None,
                x_minmax: Optional[torch.Tensor] = None, ):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        _ = r
        _ = x_minmax
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=2)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        return res


class DownBlock(Module):
    """
    ### Down block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.x_prev_channel = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

        # previous channel
        self.x_prev_channel.weight.data.fill_(0)
        self.x_prev_channel.bias.data.fill_(0)

    def forward(self,
                x: torch.Tensor,
                x_prev: torch.Tensor,
                x_minmax: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor):

        s_prev = x.shape[1]
        x = self.res(x, x_prev, x_minmax, r, t)
        s_curr = x.shape[1]
        x = self.attn(x)

        if s_prev != s_curr:
            x_prev = self.x_prev_channel(x_prev)

        return x, x_prev


class UpBlock(Module):
    """
    ### Up block
    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()
        self.x_prev_channel = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=(1, 1))

        self.x_prev_channel.weight.data.fill_(0)
        self.x_prev_channel.bias.data.fill_(0)

    def forward(self,
                x: torch.Tensor,
                x_prev: torch.Tensor,
                x_minmax: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor):
        s_prev = x.shape[1]
        x = self.res(x, x_prev, x_minmax, r, t)
        s_curr = x.shape[1]
        x = self.attn(x)

        if s_prev != s_curr:
            x_prev = self.x_prev_channel(x_prev)

        return x, x_prev


class MiddleBlock(Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self,
                x: torch.Tensor,
                x_prev: torch.Tensor,
                x_minmax: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor):
        x = self.res1(x, x_prev, x_minmax, r, t)
        x = self.attn(x)
        x = self.res2(x, x_prev, x_minmax, r, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv_x = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))
        self.conv_x_prev = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

        self.conv_x_prev.weight.data.fill_(0)
        self.conv_x_prev.bias.data.fill_(0)

    def forward(self,
                x: torch.Tensor,
                x_prev: torch.Tensor,
                x_minmax: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        _ = r
        _ = x_minmax
        x = self.conv_x(x)
        x_prev = self.conv_x_prev(x_prev)
        return x, x_prev


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv_x = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
        self.conv_x_prev = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

        self.conv_x_prev.weight.data.fill_(0)
        self.conv_x_prev.bias.data.fill_(0)

    def forward(self,
                x: torch.Tensor,
                x_prev: torch.Tensor,
                x_minmax: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        _ = r
        _ = x_minmax
        x = self.conv_x(x)
        x_prev = self.conv_x_prev(x_prev)
        return x, x_prev


class UNet(Module):
    """
    ## U-Net
    """

    def __init__(self,
                 x_val_channels: int = 1,
                 x_minmax_channels: int = 1,
                 n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `x_val_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project chunk values into feature map
        self.x_proj = nn.Conv2d(x_val_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.x_prev_proj = nn.Conv2d(x_val_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)
        self.pos_emb = TimeEmbedding(n_channels * 4)
        # +++++++++ project chunk min-max values into feature map
        self.minmax_emb_lin1 = nn.Linear(x_minmax_channels, n_channels * 4)
        self.minmax_emb_act1 = Swish()

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, x_val_channels, kernel_size=(3, 3), padding=(1, 1))

        self.attn_for_prev_val = ImageAttentionModule(in_channels=n_channels)

        # self.x_prev_proj.weight.data.fill_(0)
        # self.x_prev_proj.bias.data.fill_(0)

    def forward(self,
                x: torch.Tensor,
                x_minmax: torch.Tensor,
                x_prev: torch.Tensor,
                r: torch.Tensor,
                t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        w = get_weight_matrix(num_of_rows=x_prev.shape[2],
                              num_of_cols=x_prev.shape[3],
                              device=x.device)
        x_prev = x_prev * w

        # Get time-step embeddings
        t = self.time_emb(t)
        r = self.pos_emb(r)
        x_minmax = self.minmax_emb_lin1(x_minmax)
        x_minmax = self.minmax_emb_act1(x_minmax)

        # plt.imshow(x_prev.detach().cpu().numpy()[0, 0])
        # plt.show()

        # Get image projection
        x = self.x_proj(x)
        x_prev = self.x_prev_proj(x_prev)

        # plt.imshow(x_prev.detach().cpu().numpy()[0, 0])
        # plt.show()

        # get attention based output
        x_prev = self.attn_for_prev_val(x_prev)
        # plt.imshow(x_prev.detach().cpu().numpy()[0, 0])
        # plt.show()

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        h_prev = [x_prev]
        # First half of U-Net
        for m in self.down:
            x, x_prev = m(x, x_prev, x_minmax, r, t)
            h.append(x)
            h_prev.append(x_prev)

        # Middle (bottom)
        x = self.middle(x, x_prev, x_minmax, r, t)

        # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                x, x_prev = m(x, x_prev, x_minmax, r, t)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                s_prev = h_prev.pop()

                x = torch.cat((x, s), dim=1)
                x_prev = torch.cat((x_prev, s_prev), dim=1)
                # we need to change the x_prev channels as well here.

                x, x_prev = m(x, x_prev, x_minmax, r, t)

        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))


class ImageAttentionModule(nn.Module):
    '''
    Attention module which takes weighted previous chunk value as the input and
    give it as the output. Reverse U-net architecture contains this module.
    '''

    def __init__(self, in_channels, reduction_ratio=8):
        super(ImageAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)
        batch_size, _, height, width = x.size()

        # Compute query, key, and value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        # Compute attention scores
        energy = torch.bmm(query, key)
        attention = nn.functional.softmax(energy, dim=-1)

        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, self.in_channels, height, width)

        # Apply residual connection
        out = self.gamma * out + x

        return out
