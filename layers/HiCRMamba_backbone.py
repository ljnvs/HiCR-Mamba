__all__ = ['HiCRMambaBackbone']

from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from layers.PatchTST_layers import get_activation_fn, positional_encoding
from layers.RevIN import RevIN


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale * self.weight


class SelectivePatchMixer(nn.Module):
    """Mamba-style selective state mixer for patch tokens.

    This module is intentionally independent from the old Kimi/KDA prototype.
    It uses a diagonal selective state scan over patch tokens instead of
    Transformer attention.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 3,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.inner_dim = d_model * expand
        self.bidirectional = bidirectional

        self.in_proj = nn.Linear(d_model, self.inner_dim * 2)
        padding = max(d_conv - 1, 0) // 2
        self.dw_conv = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            kernel_size=d_conv,
            padding=padding,
            groups=self.inner_dim,
        )
        self.dt_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.b_proj = nn.Linear(self.inner_dim, d_state)
        self.c_proj = nn.Linear(self.inner_dim, d_state)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))

        if bidirectional:
            self.merge = nn.Linear(self.inner_dim * 2, self.inner_dim)
        else:
            self.merge = None
        self.out_proj = nn.Sequential(nn.Linear(self.inner_dim, d_model), nn.Dropout(dropout))

    def _conv_same_length(self, x: Tensor, seq_len: int) -> Tensor:
        x = self.dw_conv(x.transpose(1, 2)).transpose(1, 2)
        if x.size(1) > seq_len:
            x = x[:, :seq_len, :]
        elif x.size(1) < seq_len:
            x = F.pad(x, (0, 0, 0, seq_len - x.size(1)))
        return x

    def _scan(self, x: Tensor, dt: Tensor, b: Tensor, c: Tensor, reverse: bool = False) -> Tensor:
        batch_size, seq_len, _ = x.shape
        state = x.new_zeros(batch_size, self.inner_dim, self.d_state)
        outputs = []
        A = -torch.exp(self.A_log).to(dtype=x.dtype)
        positions = range(seq_len - 1, -1, -1) if reverse else range(seq_len)

        for idx in positions:
            x_t = x[:, idx, :]
            dt_t = dt[:, idx, :].unsqueeze(-1)
            b_t = b[:, idx, :].unsqueeze(1)
            c_t = c[:, idx, :].unsqueeze(1)
            decay = torch.exp(A.unsqueeze(0) * dt_t).clamp(max=1.0)
            state = state * decay + x_t.unsqueeze(-1) * b_t
            y_t = (state * c_t).sum(dim=-1) + x_t * self.D
            outputs.append(y_t)

        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.size(1)
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self._conv_same_length(u, seq_len))
        gate = torch.sigmoid(gate)

        dt = F.softplus(self.dt_proj(u)) + 1e-4
        b = self.b_proj(u)
        c = self.c_proj(u)

        y = self._scan(u, dt, b, c, reverse=False)
        if self.bidirectional:
            y_rev = self._scan(u, dt, b, c, reverse=True)
            y = self.merge(torch.cat([y, y_rev], dim=-1))

        return self.out_proj(y * gate)


class RecentStateEnhancer(nn.Module):
    def __init__(self, d_model: int, recent_k: int, dropout: float):
        super().__init__()
        self.recent_k = max(1, recent_k)
        self.summary = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        recent = x[:, -self.recent_k:, :].mean(dim=1, keepdim=True)
        recent = self.summary(recent)
        gate = self.gate(torch.cat([x, recent.expand_as(x)], dim=-1))
        return x + self.dropout(gate * recent)


class MultiScaleStateAdapter(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.coarse = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2, groups=d_model)
        self.mix = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        coarse = self.coarse(x.transpose(1, 2)).transpose(1, 2)
        if coarse.size(1) > x.size(1):
            coarse = coarse[:, :x.size(1), :]
        gate = self.mix(torch.cat([x, coarse], dim=-1))
        return x + gate * self.proj(coarse)


class ChannelControlAdapter(nn.Module):
    def __init__(self, d_model: int, n_vars: int, rank: int, dropout: float):
        super().__init__()
        rank = max(1, rank)
        self.var_embed = nn.Parameter(torch.zeros(1, n_vars, 1, d_model))
        nn.init.normal_(self.var_embed, std=0.02)
        self.gamma = nn.Sequential(nn.Linear(d_model, rank), nn.GELU(), nn.Linear(rank, d_model), nn.Tanh())
        self.beta = nn.Sequential(nn.Linear(d_model, rank), nn.GELU(), nn.Linear(rank, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        control = self.var_embed[:, :x.size(1), :, :]
        gamma = self.gamma(control)
        beta = self.beta(control)
        return x * (1.0 + 0.1 * gamma) + self.dropout(0.1 * beta)


class LiteChannelControlAdapter(nn.Module):
    def __init__(self, d_model: int, n_vars: int, rank: int, dropout: float):
        super().__init__()
        rank = max(1, rank)
        self.var_embed = nn.Parameter(torch.zeros(1, n_vars, 1, rank))
        nn.init.normal_(self.var_embed, std=0.02)
        self.gamma = nn.Sequential(nn.Linear(rank, rank), nn.GELU(), nn.Linear(rank, 1), nn.Tanh())
        self.beta = nn.Sequential(nn.Linear(rank, rank), nn.GELU(), nn.Linear(rank, 1))
        self.feature_bias = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        control = self.var_embed[:, :x.size(1), :, :]
        gamma = self.gamma(control)
        beta = self.beta(control)
        return x * (1.0 + 0.1 * gamma) + self.dropout(0.1 * beta * self.feature_bias)


class ChannelRecentAdapter(nn.Module):
    def __init__(self, d_model: int, n_vars: int, rank: int, recent_k: int, dropout: float):
        super().__init__()
        self.channel = ChannelControlAdapter(d_model, n_vars, rank, dropout)
        self.recent = RecentStateEnhancer(d_model, recent_k, dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_vars, patch_num, d_model = x.shape
        x = self.channel(x)
        x = x.reshape(batch_size * n_vars, patch_num, d_model)
        x = self.recent(x)
        return x.reshape(batch_size, n_vars, patch_num, d_model)


class TrafficRobustChannelRecentAdapter(nn.Module):
    """Conservative channel-recent adapter for highly heterogeneous sensors.

    Traffic-336 showed lower MAE but higher MSE with the full channel_recent
    adapter, suggesting fewer average errors but larger outlier errors. This
    variant keeps the same mechanism family while reducing modulation strength
    and adding a local patch smoother to damp squared-error spikes.
    """

    def __init__(self, d_model: int, n_vars: int, rank: int, recent_k: int, dropout: float):
        super().__init__()
        rank = max(1, rank)
        self.var_embed = nn.Parameter(torch.zeros(1, n_vars, 1, rank))
        nn.init.normal_(self.var_embed, std=0.02)
        self.gamma = nn.Sequential(nn.Linear(rank, rank), nn.GELU(), nn.Linear(rank, 1), nn.Tanh())
        self.beta = nn.Sequential(nn.Linear(rank, rank), nn.GELU(), nn.Linear(rank, 1), nn.Tanh())
        self.feature_scale = nn.Parameter(torch.ones(1, 1, 1, d_model))
        self.smooth = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.recent = RecentStateEnhancer(d_model, recent_k, dropout)
        self.recent_scale = nn.Parameter(torch.tensor(-1.4))
        self.smooth_scale = nn.Parameter(torch.tensor(-1.4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_vars, patch_num, d_model = x.shape
        control = self.var_embed[:, :n_vars, :, :]
        gamma = self.gamma(control)
        beta = self.beta(control) * self.feature_scale
        x = x * (1.0 + 0.05 * gamma) + self.dropout(0.05 * beta)

        flat = x.reshape(batch_size * n_vars, patch_num, d_model)
        smooth = self.smooth(flat.transpose(1, 2)).transpose(1, 2)
        flat = flat + torch.sigmoid(self.smooth_scale) * 0.1 * smooth
        recent = self.recent(flat)
        flat = flat + torch.sigmoid(self.recent_scale) * (recent - flat)
        return flat.reshape(batch_size, n_vars, patch_num, d_model)


class AdaptiveChannelRecentAdapter(nn.Module):
    """Channel gate with data-dependent recent reinforcement.

    This keeps the unified model path while allowing variables/samples that are
    sensitive to recent-state amplification to stay close to channel_gate.
    """

    def __init__(self, d_model: int, n_vars: int, rank: int, recent_k: int, dropout: float):
        super().__init__()
        self.channel = ChannelControlAdapter(d_model, n_vars, rank, dropout)
        self.recent = RecentStateEnhancer(d_model, recent_k, dropout)
        self.recent_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.gate_bias = nn.Parameter(torch.tensor(-2.0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, n_vars, patch_num, d_model = x.shape
        x = self.channel(x)
        flat = x.reshape(batch_size * n_vars, patch_num, d_model)
        recent = self.recent(flat)
        delta = recent - flat
        summary = flat.mean(dim=1, keepdim=True).expand_as(flat)
        gate = torch.sigmoid(self.recent_gate(torch.cat([flat, summary], dim=-1)) + self.gate_bias)
        flat = flat + self.dropout(gate * delta)
        return flat.reshape(batch_size, n_vars, patch_num, d_model)


class CompressedMemoryHead(nn.Module):
    def __init__(self, n_vars: int, d_model: int, patch_num: int, target_window: int, memory_slots: int, dropout: float):
        super().__init__()
        self.n_vars = n_vars
        self.memory_slots = max(1, memory_slots)
        self.slot_logits = nn.Parameter(torch.zeros(self.memory_slots, patch_num))
        nn.init.normal_(self.slot_logits, std=0.02)
        self.slot_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(d_model * self.memory_slots, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 1, 3, 2)
        weights = torch.softmax(self.slot_logits, dim=-1)
        slots = torch.einsum('bnpd,sp->bnsd', x, weights)
        slots = slots * self.slot_gate(slots)
        slots = self.flatten(slots)
        return self.dropout(self.linear(slots))


class HorizonStateHead(nn.Module):
    def __init__(self, n_vars: int, d_model: int, patch_num: int, target_window: int, dropout: float):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        self.memory_proj = nn.Linear(d_model * patch_num, d_model)
        self.horizon_embed = nn.Parameter(torch.zeros(target_window, d_model))
        nn.init.normal_(self.horizon_embed, std=0.02)
        self.readout = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        memory = x.flatten(start_dim=-2)
        memory = self.memory_proj(memory)
        horizon = self.horizon_embed.view(1, 1, self.target_window, -1)
        memory = memory.unsqueeze(2).expand(-1, -1, self.target_window, -1)
        return self.readout(torch.cat([memory, horizon.expand_as(memory)], dim=-1)).squeeze(-1)


class PatchStateBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_state: int,
        expand: int,
        d_conv: int,
        dropout: float,
        activation: str,
        bidirectional: bool,
        residual_scale: float,
    ):
        super().__init__()
        self.norm_mixer = RMSNorm(d_model)
        self.mixer = SelectivePatchMixer(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.norm_ffn = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.sigmoid(self.residual_scale)
        x = x + scale * self.mixer(self.norm_mixer(x))
        x = x + scale * self.ffn(self.norm_ffn(x))
        return x


class PatchStateEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ff: int,
        d_state: int,
        expand: int,
        d_conv: int,
        dropout: float,
        activation: str,
        bidirectional: bool,
        residual_scale: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            PatchStateBlock(
                d_model=d_model,
                d_ff=d_ff,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dropout=dropout,
                activation=activation,
                bidirectional=bidirectional,
                residual_scale=residual_scale,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class HiCRMambaBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        max_seq_len: Optional[int] = 1024,
        n_layers: int = 3,
        d_model: int = 128,
        d_ff: int = 256,
        dropout: float = 0.0,
        act: str = "gelu",
        pe: str = 'zeros',
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        padding_patch=None,
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        pm_d_state: int = 16,
        pm_expand: int = 2,
        pm_d_conv: int = 3,
        pm_bidirectional: int = 1,
        pm_residual_scale: float = 0.5,
        pm_variant: str = 'base',
        pm_memory_slots: int = 4,
        pm_recent_k: int = 3,
        pm_channel_rank: int = 8,
        **kwargs,
    ):
        del max_seq_len, fc_dropout, kwargs
        super().__init__()

        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.pm_variant = pm_variant
        patch_num = int((context_window - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = PatchStateEncoder(
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            d_state=pm_d_state,
            expand=pm_expand,
            d_conv=pm_d_conv,
            dropout=dropout,
            activation=act,
            bidirectional=bool(pm_bidirectional),
            residual_scale=pm_residual_scale,
        )
        self.post_adapter = nn.Identity()
        if pm_variant == 'recent_state':
            self.post_adapter = RecentStateEnhancer(d_model, pm_recent_k, dropout)
        elif pm_variant == 'multiscale_state':
            self.post_adapter = MultiScaleStateAdapter(d_model, dropout)
        elif pm_variant == 'channel_gate':
            self.post_adapter = ChannelControlAdapter(d_model, c_in, pm_channel_rank, dropout)
        elif pm_variant == 'channel_gate_lite':
            self.post_adapter = LiteChannelControlAdapter(d_model, c_in, pm_channel_rank, dropout)
        elif pm_variant == 'channel_recent':
            self.post_adapter = ChannelRecentAdapter(d_model, c_in, pm_channel_rank, pm_recent_k, dropout)
        elif pm_variant == 'traffic_robust':
            self.post_adapter = TrafficRobustChannelRecentAdapter(d_model, c_in, pm_channel_rank, pm_recent_k, dropout)
        elif pm_variant == 'adaptive_recent':
            self.post_adapter = AdaptiveChannelRecentAdapter(d_model, c_in, pm_channel_rank, pm_recent_k, dropout)

        self.head_nf = d_model * patch_num
        self.n_vars = c_in
        if pm_variant == 'compressed_memory':
            self.head = CompressedMemoryHead(self.n_vars, d_model, patch_num, target_window, pm_memory_slots, head_dropout)
        elif pm_variant == 'horizon_memory':
            self.head = HorizonStateHead(self.n_vars, d_model, patch_num, target_window, head_dropout)
        else:
            self.head = Flatten_Head(individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def forward(self, z: Tensor) -> Tensor:
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z = z.permute(0, 1, 3, 2)

        n_vars = z.shape[1]
        z = z.permute(0, 1, 3, 2)
        z = self.W_P(z)
        z = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3]))
        z = self.dropout(z + self.W_pos)
        z = self.encoder(z)
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        if self.pm_variant in ('channel_gate', 'channel_gate_lite', 'channel_recent', 'traffic_robust', 'adaptive_recent'):
            z = self.post_adapter(z)
        elif self.pm_variant in ('recent_state', 'multiscale_state'):
            z = torch.reshape(z, (-1, z.shape[-2], z.shape[-1]))
            z = self.post_adapter(z)
            z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        z = z.permute(0, 1, 3, 2)
        z = self.head(z)

        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z
