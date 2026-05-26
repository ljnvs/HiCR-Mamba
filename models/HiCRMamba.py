__all__ = ['HiCRMamba']

from typing import Optional

from torch import nn, Tensor

from layers.HiCRMamba_backbone import HiCRMambaBackbone
from layers.PatchTST_layers import series_decomp


class Model(nn.Module):
    """HiCR-Mamba: high-dimensional channel-recent Mamba forecasting model."""

    def __init__(
        self,
        configs,
        max_seq_len: Optional[int] = 1024,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        norm: str = 'BatchNorm',
        attn_dropout: float = 0.,
        act: str = "gelu",
        key_padding_mask: bool = 'auto',
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = 'zeros',
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type='flatten',
        verbose: bool = False,
        **kwargs,
    ):
        del d_k, d_v, norm, attn_dropout, key_padding_mask, padding_var
        del attn_mask, res_attention, pre_norm, store_attn, pretrain_head
        del head_type, verbose, kwargs
        super().__init__()

        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        n_layers = configs.e_layers
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        pm_d_state = getattr(configs, 'pm_d_state', 16)
        pm_expand = getattr(configs, 'pm_expand', 2)
        pm_d_conv = getattr(configs, 'pm_d_conv', 3)
        pm_bidirectional = getattr(configs, 'pm_bidirectional', 1)
        pm_residual_scale = getattr(configs, 'pm_residual_scale', 0.5)
        pm_variant = getattr(configs, 'pm_variant', 'base')
        pm_memory_slots = getattr(configs, 'pm_memory_slots', 4)
        pm_recent_k = getattr(configs, 'pm_recent_k', 3)
        pm_channel_rank = getattr(configs, 'pm_channel_rank', 8)

        backbone_kwargs = dict(
            c_in=c_in,
            context_window=context_window,
            target_window=target_window,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            act=act,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            pm_d_state=pm_d_state,
            pm_expand=pm_expand,
            pm_d_conv=pm_d_conv,
            pm_bidirectional=pm_bidirectional,
            pm_residual_scale=pm_residual_scale,
            pm_variant=pm_variant,
            pm_memory_slots=pm_memory_slots,
            pm_recent_k=pm_recent_k,
            pm_channel_rank=pm_channel_rank,
        )

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = HiCRMambaBackbone(**backbone_kwargs)
            self.model_res = HiCRMambaBackbone(**backbone_kwargs)
        else:
            self.model = HiCRMambaBackbone(**backbone_kwargs)

    def forward(self, x):
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)
        return x
