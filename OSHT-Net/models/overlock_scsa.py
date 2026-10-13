"""
OSHT-Net backbone: OverLoCK-SCSA with Context Bridge SCSA.

Architecture overview (Sec. 3 of the paper):
    Base-Net      : 4-stage pyramid of SCSA-Enhanced Basic Blocks (RepConvBlock)
    Overview-Net  : 1x1 projection of the Stage-4 feature into an 84-dim context
    Focus-Net     : Dynamic Blocks (ContMix) guided by the context, with a
                    Context Bridge-SCSA module at the entry of each sub-stage

Reference model:
    overlock_xt_bridge_scsa  -> OverLoCK-SCSA (Context Bridge SCSA), used as the
                                comp / ssl feature extractor in OSHT-Net.

The implementation is based on the official OverLoCK codebase
(https://github.com/LMMMEng/OverLoCK).
"""
import torch
import timm
import torch.distributed
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from natten.functional import na2d_av
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model

from .SCSA import SCSA, BasicBlockSCSA


# --------------------------------------------------------------------------
# Basic operators
# --------------------------------------------------------------------------

def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation,
               groups, bias, attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = (kernel_size[0] == kernel_size[1] and kernel_size[0] > 5
                       and padding == (kernel_size[0] // 2, kernel_size[1] // 2))

    if attempt_use_lk_impl and need_large_impl:
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        except Exception:
            DepthWiseConv2dImplicitGEMM = None
        if (DepthWiseConv2dImplicitGEMM is not None and need_large_impl
                and in_channels == out_channels and out_channels == groups
                and stride == 1 and dilation == 1):
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)

    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    return nn.SyncBatchNorm(dim) if use_sync_bn else nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return (conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1),
            bn.bias + (conv_bias - bn.running_mean) * bn.weight / std)


def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:  # depthwise kernel
        return F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
    slices = []
    for i in range(kernel.size(1)):
        dilated = F.conv_transpose2d(kernel[:, i:i + 1, :, :], identity_kernel, stride=dilate_rate)
        slices.append(dilated)
    return torch.cat(slices, dim=1)


def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    return large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim // 2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim // 2),
        nn.GELU(),
        nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
    )


def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )


# --------------------------------------------------------------------------
# Building components
# --------------------------------------------------------------------------

class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.GELU, out_act=nn.Sigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            inner_act(),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            out_act(),
        )

    def forward(self, x):
        return x * self.proj(x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x.contiguous()


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2)."""

    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(-1, -2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        return (self.gamma * Nx + 1) * x


class DilatedReparamBlock(nn.Module):
    """Dilated Reparam Block (UniRepLKNet)."""

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                     padding=kernel_size // 2, dilation=1, groups=channels,
                                     bias=deploy, attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        if kernel_size == 17:
            self.kernel_sizes = [5, 7, 9, 3, 3, 3]
            self.dilates = [1, 1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 7, 3, 3, 3]
            self.dilates = [1, 1, 2, 3, 4, 5]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3, 3]
            self.dilates = [1, 1, 2, 3]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__(f'dil_conv_k{k}_{r}',
                                 nn.Conv2d(channels, channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r,
                                           groups=channels, bias=False))
                self.__setattr__(f'dil_bn_k{k}_{r}', get_bn(channels, use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__(f'dil_conv_k{k}_{r}')
            bn = self.__getattr__(f'dil_bn_k{k}_{r}')
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__(f'dil_conv_k{k}_{r}')
                bn = self.__getattr__(f'dil_bn_k{k}_{r}')
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2),
                                     stride=1, padding=origin_k.size(2) // 2, dilation=1,
                                     groups=origin_k.size(0), bias=True,
                                     attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__(f'dil_conv_k{k}_{r}')
                self.__delattr__(f'dil_bn_k{k}_{r}')


class CTXDownsample(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim))
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim // 4, h_dim // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h_dim // 4))

    def forward(self, x, ctx):
        return self.x_proj(x), self.h_proj(ctx)


class ResDWConv(nn.Conv2d):
    """Depthwise convolution with a residual connection."""

    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)

    def forward(self, x):
        return x + super().forward(x)


class RepConvBlock(nn.Module):
    """SCSA-Enhanced Basic Block used in all four Base-Net stages.

    Convolutional main path (inherited from OverLoCK) plus one appended
    SCSA Residual Block (``BasicBlockSCSA``).
    """

    def __init__(self, dim=64, kernel_size=7, mlp_ratio=4, shortcut_scale=1,
                 ls_init_value=None, res_scale=False, drop_path=0,
                 norm_layer=LayerNorm2d, use_gemm=False, deploy=False, use_checkpoint=False):
        super().__init__()
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint

        mlp_dim = int(dim * mlp_ratio)
        self.dwconv = ResDWConv(dim, kernel_size=3)
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy,
                                use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )
        self.ls = LayerScale(dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.scsa_basicblock = BasicBlockSCSA(in_channels=dim, out_channels=dim,
                                              scsa_cfg=dict(head_num=4))
        self.scsa_identity = nn.Identity()
        self.shortcut_scale = shortcut_scale

    def forward_features(self, x):
        x = self.dwconv(x)
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))
        x = self.scsa_identity(x) + self.scsa_basicblock(x) * self.shortcut_scale
        return x

    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        return x


class DynamicConvBlock(nn.Module):
    """ContMix dynamic convolution block (token mixer of Focus-Net)."""

    def __init__(self, dim=64, ctx_dim=32, kernel_size=7, smk_size=5, num_heads=2,
                 mlp_ratio=4, ls_init_value=None, res_scale=False, drop_path=0,
                 norm_layer=LayerNorm2d, is_first=False, is_last=False,
                 use_gemm=False, deploy=False, use_checkpoint=False, **kwargs):
        super().__init__()
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim * mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1)
            self.h_scale = LayerScale(ctx_dim, init_value=1)

        self.dwconv1 = ResDWConv(out_dim, kernel_size=3)
        self.norm1 = norm_layer(out_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )
        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // 2),
        )
        self.weight_proj = nn.Conv2d(49, kernel_size ** 2 + smk_size ** 2, kernel_size=1)
        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy,
                                use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )
        self.se_layer = SEModule(dim)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )
        self.ls1 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.get_rpb()

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size)
        idx_w = torch.arange(0, kernel_size)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return idx_h, idx_w, idx_k

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long)
        num_repeat_h[kernel_size // 2] = height - (kernel_size - 1)
        num_repeat_w[kernel_size // 2] = width - (kernel_size - 1)
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2 * kernel_size - 1)) \
                  + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size ** 2))
        bias_idx = torch.flip(bias_idx, [0])
        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        rpb = rpb.reshape(1, int(self.num_heads), int(height), int(width), int(kernel_size ** 2))
        return attn + rpb

    def _forward_inner(self, x, h_x, h_r):
        input_resolution = x.shape[2:]
        B, C, H, W = x.shape
        B, C_h, H_h, W_h = h_x.shape

        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f = self.norm1(x_f)
        x = self.fusion(x_f)
        gate = self.gate(x)
        lepe = self.lepe(x)

        is_pad = False
        if min(H, W) < self.kernel_size:
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            x_f = F.interpolate(x_f, size=size, mode='bilinear', align_corners=False)
            H, W = size

        query, key = torch.split(x_f, split_size_or_sections=[C, C_h], dim=1)
        query = self.weight_query(query) * self.scale
        key = self.weight_key(key)
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1, attn2 = torch.split(weight, [self.smk_size ** 2, self.kernel_size ** 2], dim=-1)
        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)

        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)

        x = torch.cat([x1, x2], dim=1)
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)

        if is_pad:
            x = F.adaptive_avg_pool2d(x, input_resolution)

        x = self.dyconv_proj(x)
        x = x + lepe
        x = self.se_layer(x)
        x = gate * x
        x = self.proj(x)

        if self.res_scale:
            x = self.ls1(identity) + self.drop_path(x)
        else:
            x = identity + self.drop_path(self.ls1(x))

        x = self.dwconv2(x)
        if self.res_scale:
            x = self.ls2(x) + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))

        if self.is_last:
            return x, None
        l_x, h_x = torch.split(x, [C, C_h], dim=1)
        return l_x, h_x

    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        return self._forward_inner(x, h_x, h_r)


# --------------------------------------------------------------------------
# Backbone assembly
# --------------------------------------------------------------------------

class OverLoCK_SCSA_BasicBlock(nn.Module):
    """OverLoCK backbone with an SCSA Residual Block appended to every Basic Block."""

    def __init__(self, depth=[2, 2, 3, 2], sub_depth=[6, 2], in_chans=3,
                 embed_dim=[56, 112, 256, 336], kernel_size=[17, 15, 13, 7],
                 mlp_ratio=[4, 4, 4, 4], sub_mlp_ratio=[3, 3], sub_num_heads=[4, 6],
                 ls_init_value=[None, None, 1, 1], res_scale=True, smk_size=5,
                 deploy=False, use_gemm=True, use_ds=True, drop_rate=0, drop_path_rate=0,
                 norm_layer=LayerNorm2d, projection=1024, num_classes=1000,
                 use_checkpoint=[0, 0, 0, 0], shortcut_scale=1):
        super().__init__()
        fusion_dim = embed_dim[-1] + embed_dim[-1] // 4
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv2d(embed_dim[-1], embed_dim[-1] // 4, kernel_size=1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()
        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()

        for i in range(depth[0]):
            self.blocks1.append(RepConvBlock(
                dim=embed_dim[0], kernel_size=kernel_size[0], mlp_ratio=mlp_ratio[0],
                ls_init_value=ls_init_value[0], res_scale=res_scale, drop_path=dpr[i],
                norm_layer=norm_layer, use_gemm=use_gemm, deploy=deploy,
                use_checkpoint=(i < use_checkpoint[0]), shortcut_scale=shortcut_scale))
        for i in range(depth[1]):
            self.blocks2.append(RepConvBlock(
                dim=embed_dim[1], kernel_size=kernel_size[1], mlp_ratio=mlp_ratio[1],
                ls_init_value=ls_init_value[1], res_scale=res_scale, drop_path=dpr[i + depth[0]],
                norm_layer=norm_layer, use_gemm=use_gemm, deploy=deploy,
                use_checkpoint=(i < use_checkpoint[1]), shortcut_scale=shortcut_scale))
        for i in range(depth[2]):
            self.blocks3.append(RepConvBlock(
                dim=embed_dim[2], kernel_size=kernel_size[2], mlp_ratio=mlp_ratio[2],
                ls_init_value=ls_init_value[2], res_scale=res_scale, drop_path=dpr[i + sum(depth[:2])],
                norm_layer=norm_layer, use_gemm=use_gemm, deploy=deploy,
                use_checkpoint=(i < use_checkpoint[2]), shortcut_scale=shortcut_scale))
        for i in range(depth[3]):
            self.blocks4.append(RepConvBlock(
                dim=embed_dim[3], kernel_size=kernel_size[3], mlp_ratio=mlp_ratio[3],
                ls_init_value=ls_init_value[3], res_scale=res_scale, drop_path=dpr[i + sum(depth[:3])],
                norm_layer=norm_layer, use_gemm=use_gemm, deploy=deploy,
                use_checkpoint=(i < use_checkpoint[3]), shortcut_scale=shortcut_scale))
        for i in range(sub_depth[0]):
            self.sub_blocks3.append(DynamicConvBlock(
                dim=embed_dim[2], ctx_dim=embed_dim[-1], kernel_size=kernel_size[2],
                num_heads=sub_num_heads[0], pool_size=7, mlp_ratio=sub_mlp_ratio[0],
                ls_init_value=ls_init_value[2], res_scale=res_scale,
                drop_path=dpr[i + sum(depth)], norm_layer=norm_layer, smk_size=smk_size,
                use_gemm=use_gemm, deploy=deploy, is_first=(i == 0),
                use_checkpoint=(i < use_checkpoint[2])))
        for i in range(sub_depth[1]):
            self.sub_blocks4.append(DynamicConvBlock(
                dim=embed_dim[3], ctx_dim=embed_dim[-1], kernel_size=kernel_size[-1],
                num_heads=sub_num_heads[1], pool_size=7, mlp_ratio=sub_mlp_ratio[1],
                ls_init_value=ls_init_value[3], res_scale=res_scale,
                drop_path=dpr[i + sum(depth) + sub_depth[0]], norm_layer=norm_layer,
                smk_size=smk_size, is_first=False, is_last=(i == sub_depth[1] - 1),
                use_gemm=use_gemm, deploy=deploy, use_checkpoint=(i < use_checkpoint[3])))

        if use_ds:
            self.aux_head = nn.Sequential(
                nn.BatchNorm2d(embed_dim[-1]),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim[-1], num_classes, kernel_size=1) if num_classes > 0 else nn.Identity())

        self.head = nn.Sequential(
            nn.Conv2d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def reparam(self):
        for m in self.modules():
            if isinstance(m, DilatedReparamBlock):
                m.merge_dilated_branches()

    def forward_pre_features(self, x):
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        return x

    def forward_base_features(self, x):
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)
        return x, ctx

    def forward_sub_features(self, x, ctx):
        ctx_cls = ctx
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode='bilinear', align_corners=False)
        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)
        x, ctx = self.patch_embedx(x, ctx)
        for blk in self.sub_blocks4:
            x, ctx = blk(x, ctx, ctx_ori)
        return x, ctx_cls

    def forward_features(self, x):
        x = self.forward_pre_features(x)
        x, ctx = self.forward_base_features(x)
        x, ctx_cls = self.forward_sub_features(x, ctx)
        return x, ctx_cls

    def forward(self, x):
        x, ctx = self.forward_features(x)
        x = self.head(x).flatten(1)
        if hasattr(self, 'aux_head') and self.training:
            ctx = self.aux_head(ctx).flatten(1)
            return dict(main=x, aux=ctx)
        return x


class OverLoCK_SCSA_BasicBlock_ContextBridgeSCSA(OverLoCK_SCSA_BasicBlock):
    """OverLoCK-SCSA with Context Bridge-SCSA (the OSHT-Net backbone).

    At the entry of each Focus-Net sub-stage, the local stream and the context
    stream are concatenated, refined by an SCSA block with a residual connection,
    and split back into the two streams (Sec. 3.2, "Context Bridge-SCSA").
    """

    def __init__(self, bridge_head_nums=[4, 6], **kwargs):
        super().__init__(**kwargs)
        dim3 = self.embed_dim[2] + self.embed_dim[-1] // 4   # 256 + 84 = 340
        dim4 = self.embed_dim[3] + self.embed_dim[-1] // 4   # 336 + 84 = 420
        self.bridge_scsa3 = SCSA(dim=dim3, head_num=bridge_head_nums[0])
        self.bridge_scsa4 = SCSA(dim=dim4, head_num=bridge_head_nums[1])

    def forward_sub_features(self, x, ctx):
        ctx_cls = ctx
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Stage-3 bridge: concat -> SCSA -> residual -> split
        fused_x3 = torch.cat([x, ctx_up], dim=1)
        fused_x3 = fused_x3 + self.bridge_scsa3(fused_x3)
        x, ctx_up = torch.split(fused_x3, [x.shape[1], ctx_up.shape[1]], dim=1)

        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)

        x, ctx = self.patch_embedx(x, ctx)

        # Stage-4 bridge: concat -> SCSA -> residual -> split
        fused_x4 = torch.cat([x, ctx_ori], dim=1)
        fused_x4 = fused_x4 + self.bridge_scsa4(fused_x4)
        x, ctx_ori = torch.split(fused_x4, [x.shape[1], ctx_ori.shape[1]], dim=1)

        for blk in self.sub_blocks4:
            x, ctx = blk(x, ctx, ctx_ori)
        return x, ctx_cls


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------

def _cfg(crop_pct=0.925):
    return {
        'crop_pct': crop_pct,
        'input_size': (3, 224, 224),
        'interpolation': 'bicubic',
        'mean': timm.data.IMAGENET_DEFAULT_MEAN,
        'std': timm.data.IMAGENET_DEFAULT_STD,
    }


@register_model
def overlock_xt_bridge_scsa(pretrained=False, pretrained_cfg=None, **kwargs):
    """OSHT-Net backbone (OverLoCK-XT + Block SCSA + Context Bridge SCSA)."""
    for k in ('pretrained_cfg_overlay', 'cache_dir', 'features_only', 'out_indices'):
        kwargs.pop(k, None)
    model = OverLoCK_SCSA_BasicBlock_ContextBridgeSCSA(
        depth=[2, 2, 3, 2],
        sub_depth=[6, 2],
        embed_dim=[56, 112, 256, 336],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 6],
        sub_mlp_ratio=[3, 3],
        bridge_head_nums=[4, 6],
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.925)
    if pretrained:
        # ImageNet-1K checkpoint of the plain OverLoCK-XT; strict=False because
        # the SCSA / bridge modules are randomly initialized.
        from timm.models import load_checkpoint
        load_checkpoint(model, 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth', strict=False)
    return model
