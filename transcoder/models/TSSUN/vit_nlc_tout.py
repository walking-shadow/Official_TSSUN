# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
import os
import torch
import numpy as np
from functools import partial
from dict_recursive_update import recursive_update
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
cwd = os.getcwd()
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    ATTENTION_MODE = 'flash'
except:
    ATTENTION_MODE = 'math'
from transcoder.models.TSSUN.wave_dynamic_layer_new import Dynamic_MLP_OFA_Encoder, Dynamic_MLP_OFA_Decoder

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, window_size=None, rel_pos_spatial=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial = rel_pos_spatial
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.window_size = window_size
        if COMPAT:
            if COMPAT == 2:
                self.rel_pos_h = nn.Parameter(torch.zeros(2 * window_size[0] - 1, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(2 * window_size[1] - 1, head_dim))
            else:
                q_size = window_size[0]
                kv_size = q_size
                rel_sp_dim = 2 * q_size - 1
                self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
                self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x)
        
        if ATTENTION_MODE == 'math':
            qkv = rearrange(qkv, 'B N (K H D) -> K B H N D', K=3, H=self.num_heads)
            q, k, v = qkv.unbind(0) # make torchscript happy (cannot use tensor as tuple)   --> (batchsize, heads, len, head_dim)
            attn = ((q * self.scale) @ k.transpose(-2, -1))
            attn = torch.softmax(attn, dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # =====================================
        elif ATTENTION_MODE == 'flash':
            qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
            data_type = qkv.dtype
            qkv=qkv.to(torch.float16) 
            x = flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=self.scale, causal=False).reshape(B, N, C)    
            x=x.to(data_type)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.

    Source: https://github.com/facebookresearch/mvit/
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, rel_pos_spatial=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial=rel_pos_spatial

        if COMPAT:
            q_size = window_size[0]
            kv_size = window_size[1]
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        x = x.reshape(B_, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)  # num_Windows*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # num_Windows*B, window_size*window_size, C

        B_w = x.shape[0]
        N_w = x.shape[1]
        
        qkv = self.qkv(x).reshape(B_w, N_w, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)   --> (batchsize, heads, len, head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
            
        x = self.proj(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)

        return x


# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  window_size=None, window=False, rel_pos_spatial=False):
#         super().__init__()

#         self.time_norm1 = norm_layer(dim)
#         self.time_attn = Attention(
#             dim, num_heads=num_heads, qkv_bias=qkv_bias,
#             window_size=window_size, rel_pos_spatial=rel_pos_spatial)
#         self.time_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.time_norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.time_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

#         self.norm1 = norm_layer(dim)
#         if not window:
#             self.attn = Attention(
#                 dim, num_heads=num_heads, qkv_bias=qkv_bias,
#                 window_size=window_size, rel_pos_spatial=rel_pos_spatial)
#         else:
#             self.attn = WindowAttention(
#                 dim, num_heads=num_heads, qkv_bias=qkv_bias,
#                 window_size=window_size, rel_pos_spatial=rel_pos_spatial,
#             )
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
    
       
#         # self.mod_norm1 = norm_layer(dim)
#         # self.mod_attn = Attention(
#         #     dim, num_heads=num_heads, qkv_bias=qkv_bias,
#         #     window_size=window_size, rel_pos_spatial=rel_pos_spatial)
#         # self.mod_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # self.mod_norm2 = norm_layer(dim)
#         # self.mod_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
       

#     def forward(self, x, H, W, mask=None):
#         # x shape: b t l c
#         b,t,l,c = x.shape

#         x = rearrange(x, 'b t l c -> (b l) t c')
#         # print(f'x.shape:{x.shape}')
#         x = x + self.time_drop_path(self.time_attn(self.time_norm1(x), H, W))
#         x = x + self.time_drop_path(self.time_mlp(self.time_norm2(x)))

#         x = rearrange(x, '(b l) t c -> (b t) l c', b=b,l=l)
#         # print(f'x.shape:{x.shape}')

#         x = x + self.drop_path(self.attn(self.norm1(x), H, W))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
        
#         x = rearrange(x, '(b t) l c -> b t l c', b=b)
#         # print(f'x.shape:{x.shape}')

#         # x = x + self.mod_drop_path(self.mod_attn(self.mod_norm1(x), H, W))
#         # x = x + self.mod_drop_path(self.mod_mlp(self.mod_norm2(x)))

#         # x = rearrange(x, '(b t l) c -> b t l c', b=b,t=t)


#         # x = rearrange(x, 'b t l c -> (b t) l c', b=b,l=l)
#         # print(f'x.shape:{x.shape}')

#         # x = x + self.drop_path(self.attn(self.norm1(x), H, W))
#         # x = x + self.drop_path(self.mlp(self.norm2(x)))
        
#         # x = rearrange(x, '(b t) l c -> b t l c', b=b,m=m)

#         return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, window=False, rel_pos_spatial=False):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial)
        else:
            self.attn = WindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias,
                window_size=window_size, rel_pos_spatial=rel_pos_spatial,
            )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,),
        )

    def forward(self, x, H, W, mask=None):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = x + self.conv_layer(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16,patch_stride=16,patch_padding=2, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_shape = ((img_size[0]-patch_size[0]+2*patch_padding[0]) // patch_stride[0] + 1, \
                                   (img_size[1]-patch_size[1]+2*patch_padding[1]) // patch_stride[1] + 1)  # could be dynamic
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]  # could be dynamic
        self.patch_size = patch_size


        # self.proj = nn.Linear(in_chans, embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_padding,)



    def forward(self, x, mask=None, **kwargs):
        # x = self.proj_dofa(x)
        x = self.proj(x)  # blc
        # Hp, Wp = x.shape[2], x.shape[3]
        # x = x.flatten(2).transpose(1, 2)
        # # print(f'PatchEmbed x shape:{x.shape}')

        # if mask is not None:
        #     mask = F.interpolate(mask[None].float(), size=(Hp, Wp)).to(torch.bool)[0]

        # return x, (Hp, Wp), mask
        return x


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class ViT_Encoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, 
                 img_size=224,
                 patch_size=16, 
                 patch_stride=16,
                 patch_padding=2,
                 in_chans=32, 
                 out_chans=32,
                 z_dim=None,
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4.,
                 qkv_bias=False, 
                 window_size=(14,14),
                 drop_path_rate=0.,
                 norm_layer=None,
                 window=True,
                 use_abs_pos_emb=False,
                 interval=3, 
                 bn_group=None, 
                 test_pos_mode='simple_interpolate',
                 learnable_pos=False, 
                 rel_pos_spatial=False, 
                 lms_checkpoint_train=False, 
                 pad_attn_mask=False, 
                 freeze_iters=0,
                 act_layer='GELU', 
                 pre_ln=False, 
                 mask_input=False, 
                 ending_norm=True,
                 round_padding=False, 
                 compat=False,
                 enc_dim=192,
                 unify_time_length=2,
                 ):
        super().__init__()
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding
        self.patch_size = patch_size
        self.img_size = img_size
        self.depth = depth
        self.num_heads =num_heads
        self.Hp, self.Wp = img_size[0], img_size[1]
        self.ori_Hp, self.ori_Hw = (img_size[0]-patch_size[0]+2*patch_padding[0]) // patch_stride[0] + 1, \
                                   (img_size[1]-patch_size[1]+2*patch_padding[1]) // patch_stride[1] + 1
        # self.unify_time_length = unify_time_length
        self.in_chans = in_chans
        self.out_chans = out_chans
        

        global COMPAT
        COMPAT = compat

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models


        self.tss_unify = Dynamic_MLP_OFA_Encoder(in_chans, in_chans)

        # self.tss_unify = nn.Linear(3*1*2*2, 64)  # for whu
        # self.tss_unify = nn.Linear(3*2*4*4, 32*2)  # for levircd
        # self.tss_unify = nn.Linear(3*4*4, 64)  # for whucd
        # self.tss_unify = nn.Linear(3*4*1*1, 32*2)  # for tscd

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, patch_stride= patch_stride, patch_padding=patch_padding,
        #     in_chans=in_chans, embed_dim=embed_dim)

        # self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_padding,)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, in_chans*2, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(in_chans*2),
            nn.SiLU(),
            nn.Conv2d(in_chans*2, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_padding,),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1,),
        )


        num_patches = self.ori_Hp*self.ori_Hw

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=learnable_pos)  # 包括空间/模态/时间的pos_embedding
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.ori_Hp, self.ori_Hw), cls_token=False)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.pos_embed.data.copy_(pos_embed)
        else:
            raise
        # self.time_embed = nn.Parameter(torch.randn(1, unify_time_length, embed_dim))  # 初始化为标准正态分布
        self.task_embedding = nn.Embedding(4, embed_dim)  # 用于区分输入的4个数据集和对应的任务
        self.data_task_info = {
            'whu':{'task_index': 0,'input_time': 1, 'out_time': 1, 'out_channel':1, 'out_size': 512},
            'whucd':{'task_index': 1, 'input_time': 2, 'out_time': 2, 'out_channel':1, 'out_size': 1024},
            'levircd':{'task_index': 2, 'input_time': 2, 'out_time': 1, 'out_channel':1, 'out_size': 1024},
            'tscd':{'task_index': 3, 'input_time': 4, 'out_time': 3, 'out_channel':1, 'out_size': 256},
        }

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert interval == len(window_size)+1, 'interval must be equal to len(window_size)+1'
        self.interval = interval

        self.blocks = nn.ModuleList()
        for i in range(0, depth//2):
            which_win = min(i%interval, len(window_size)-1)  # min(i%4, 2)
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=window_size[which_win] if ((i + 1) % interval != 0) else (self.ori_Hp, self.ori_Hw),  # 选择三种窗口形状，然后选择全图注意力
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            self.blocks.append(block)

        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only



        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()
        ###
        self.test_pos_mode = test_pos_mode
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if self.mask_input else None

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    

    def embedding_forward(self, input_tensor, dataset_name, **kwargs):
        # input_tensor shape: b,t,c,h,w
        # b,t,c,h,w = input_tensor.shape
        # 这个函数把t维度和B维度合并在一起，然后进行tss_unify
        # print(f'embedding_forward input_tensor shape:{input_tensor.shape}')

        b,t,c,h,w = input_tensor.shape
        input_tensor = rearrange(input_tensor, 'b t c h w -> (b t) c h w')
        # print(f'embedding_forward input_tensor shape after rearrange:{input_tensor.shape}')
        b,c,h,w = input_tensor.shape

        patch_size = h//self.img_size[0]
        assert h==w, 'h!=w'
        # 使用 unfold 提取 patch
        patches = input_tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # (B, C, H/patch, W/patch, patch, patch)
        # 调整维度顺序以匹配标准的 (B, num_patches, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        input_tensor = rearrange(patches, 'b h w c p q -> b (h w) (c p q)')  # 把空间压缩维度、时间维度都叠在channel维度上
        # print(f'embedding_forward input_tensor shape after unfold:{input_tensor.shape}')
        tss_metadata = torch.arange(1,patch_size*patch_size+1).repeat(c).to(input_tensor.device, input_tensor.dtype).unsqueeze(0).unsqueeze(0).repeat(b,t,1)  # b,t,l
        tss_metadata = rearrange(tss_metadata, 'b t l -> (b t) l')

        input_tensor = self.tss_unify(input_tensor, tss_metadata)  # ZSJ
        # input_tensor = self.tss_unify(input_tensor)

        # print(f'embedding_forward input_tensor shape after tss_unify:{input_tensor.shape}')
        # # input_tensor = self.tss_unify(input_tensor)
        # # input_tensor = rearrange(input_tensor, 'b l (t c) -> b t l c', t=self.unify_time_length)
        # print(f'embedding_forward input_tensor after tss_unify min max:{input_tensor.min()}, {input_tensor.max()}')

        input_tensor = rearrange(input_tensor, 'b (h w) c -> b c h w', h=self.img_size[0], w=self.img_size[1])
        input_tensor = self.patch_embed(input_tensor)
        # print(f'embedding_forward input_tensor shape after patch_embed:{input_tensor.shape}')
        input_tensor = rearrange(input_tensor, 'b c h w -> b (h w) c', b=b)
        # print(f'embedding_forward input_tensor after patch_embed min max:{input_tensor.min()}, {input_tensor.max()}')

        # all_pos_embed =  self.pos_embed.unsqueeze(0) + self.time_embed.unsqueeze(-2)
        all_pos_embed =  self.pos_embed.unsqueeze(0)
        all_pos_embed = rearrange(all_pos_embed, 'b t l c -> (b t) l c').repeat(b,1,1)

        b,l,c = input_tensor.shape
        task_embedding_index = torch.tensor(self.data_task_info[dataset_name]['task_index']).repeat(b).to(device=input_tensor.device, dtype=torch.long)
        task_embedding = self.task_embedding(task_embedding_index)  # b,c
        task_embedding = rearrange(task_embedding, 'b (l c) -> b l c', l=1, c=c).repeat(1,l,1)


        input_tensor = input_tensor + all_pos_embed + task_embedding     #get_abs_pos(pos_embed, False, (self.ori_Hp, self.ori_Hw), patch_shape)
        # print(f'embedding_forward input_tensor final min max:{input_tensor.min()}, {input_tensor.max()}')
        # print(f'embedding_forward input_tensor final shape:{input_tensor.shape}')

        return input_tensor
    
    def encoder_forward(self, x):

        # x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity
        # for i in range(len(self.blocks)-1):
        for i in range(len(self.blocks)):

            # if i == len(self.blocks)-2:
            #     mean = self.blocks[i](x, self.Hp, self.Wp)  # 不需要mean和var，正常输出即可
            #     logvar = self.blocks[i+1](x, self.Hp, self.Wp)
            #     x = torch.cat([mean,logvar], 2)

            #     return x  #  we here control the last 2 blocks to genertate 2*dimensions' data.
            # else:
            #     x = self.blocks[i](x, self.Hp, self.Wp)
            # if (i+1) % (self.interval+1) == 0:
            #     # 使用时间维度的注意力
            #     x = rearrange(x, 'b t l c -> b l t c')

            x = self.blocks[i](x, self.ori_Hp, self.ori_Hw)
        # print(f'block x shape:{x.shape}')

            # if (i+1) % (self.interval+1) == 0:
            #     # 使用时间维度的注意力
            #     x = rearrange(x, 'b l t c -> b t l c')
        # print(f'encoder_forward input_tensor final min max:{x.min()}, {x.max()}')

        return x

    def forward(self, input_tensor, dataset_name, **kwargs):
        # era5_latent_data shape: b,l,c
        # print(f'input_var x shape:{input_var.shape}')
        dataset_name = dataset_name[0]

        x = self.embedding_forward(input_tensor, dataset_name, **kwargs)  # b,m,l,c
        # print(f'embedding_forward x shape:{x.shape}')
        x = self.encoder_forward(x)
        # print(f'encoder_forward x shape:{x.shape}')



        # B ,N ,C = x.shape
        # x = x.reshape(B, self.Hp, self.Wp, C).permute(0,3,1,2)

        return x

class ViT_Decoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                img_size=224, 
                patch_size=16, 
                patch_stride=16, 
                patch_padding=2,
                in_chans=32, 
                out_chans=32, 
                z_dim=None,
                embed_dim=768, 
                depth=12,
                num_heads=12, 
                mlp_ratio=4., 
                qkv_bias=False, 
                window_size=(14,14), 
                drop_path_rate=0., 
                norm_layer=None, 
                window=True,
                use_abs_pos_emb=False, 
                interval=3, 
                bn_group=None, 
                test_pos_mode='simple_interpolate',
                learnable_pos=False, 
                rel_pos_spatial=False, 
                lms_checkpoint_train=False,
                pad_attn_mask=False, 
                freeze_iters=0,
                act_layer='GELU', 
                pre_ln=False, 
                mask_input=False, 
                ending_norm=True,
                round_padding=False, 
                compat=False,
                dec_dim = 192,
                unify_time_length=2,
                ):
        super().__init__()
        self.pad_attn_mask = pad_attn_mask  # only effective for detection task input w/ NestedTensor wrapping
        self.lms_checkpoint_train = lms_checkpoint_train
        self.freeze_iters = freeze_iters
        self.mask_input = mask_input
        self.ending_norm = ending_norm
        self.round_padding = round_padding
        self.patch_size = patch_size
        self.img_size = img_size
        self.depth = depth
        self.num_heads =num_heads
        self.Hp, self.Wp = (img_size[0]-patch_size[0]+2*patch_padding[0]) // patch_stride[0] + 1, \
                                   (img_size[1]-patch_size[1]+2*patch_padding[1]) // patch_stride[1] + 1
        
        self.dec_dim = dec_dim
        self.out_chans = out_chans
        self.interval = interval

        global COMPAT
        COMPAT = compat
        self.patch_shape = (self.Hp, self.Wp)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule


        self.blocks = nn.ModuleList()
        for i in range(depth//2, depth):
            which_win = min(i%interval, len(window_size)-1)
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop_path=dpr[i], norm_layer=norm_layer,
                window_size=window_size[which_win] if ((i + 1) % interval != 0) else self.patch_shape,
                window=((i + 1) % interval != 0) if window else False,
                rel_pos_spatial=rel_pos_spatial,
                act_layer=QuickGELU if act_layer == 'QuickGELU' else nn.GELU
            )
            self.blocks.append(block)


        self.ln_pre = norm_layer(embed_dim) if pre_ln else nn.Identity()  # for clip model only
        self.norm = norm_layer(embed_dim)


        # self.final = nn.Sequential(
        #             nn.Linear(embed_dim, embed_dim*2),
        #             nn.GELU(),
        #             nn.Linear(embed_dim*2, out_chans)  # 进行二通道二分类设置
        #         )
        self.final_linear = nn.Linear(embed_dim, embed_dim)

        # self.temporal_unify_encoder = Dynamic_MLP_OFA_Encoder(unify_time_length*embed_dim, unify_time_length*embed_dim)
        # # self.temporal_unify_linear = nn.Linear(unify_time_length*embed_dim, unify_time_length*embed_dim)
        # self.temporal_unify_decoder = Dynamic_MLP_OFA_Decoder(unify_time_length*embed_dim, unify_time_length*embed_dim, decoder_bias=True)

        self.temporal_unify_encoder = Dynamic_MLP_OFA_Encoder(unify_time_length, unify_time_length)
        # # self.temporal_unify_linear = nn.Linear(unify_time_length*embed_dim, unify_time_length*embed_dim)
        self.temporal_unify_decoder = Dynamic_MLP_OFA_Decoder(unify_time_length, unify_time_length, decoder_bias=True)

        # # for whu
        # self.temporal_unify_encoder = nn.Linear(1, unify_time_length)
        # # self.temporal_unify_linear = nn.Linear(unify_time_length*embed_dim, unify_time_length*embed_dim)
        # self.temporal_unify_decoder = nn.Linear(unify_time_length, 1)

        # for whucd
        # self.temporal_unify_encoder = nn.Linear(2*embed_dim, unify_time_length*embed_dim)
        # self.temporal_unify_linear = nn.Linear(unify_time_length*embed_dim, unify_time_length*embed_dim)
        # self.temporal_unify_decoder = nn.Linear(unify_time_length*embed_dim, 2*embed_dim)


        # self.final = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_stride, padding=patch_padding,)
        self.final = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1,),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
            nn.ConvTranspose2d(embed_dim, out_chans*2, kernel_size=patch_size, stride=patch_stride, padding=patch_padding,),
            nn.BatchNorm2d(out_chans*2),
            nn.SiLU(),
            nn.Conv2d(out_chans*2, out_chans, kernel_size=3, stride=1, padding=1,),
        )
        # self.final = nn.Linear(embed_dim, out_chans)

        self.task_embedding = nn.Embedding(4, embed_dim)  # 用于区分输入的4个数据集和对应的任务
        self.data_task_info = {
            'whu':{'task_index': 0,'input_time': 1, 'out_time': 1, 'out_channel':1, 'out_size': 512},
            'whucd':{'task_index': 1, 'input_time': 2, 'out_time': 2, 'out_channel':1, 'out_size': 1024},
            'levircd':{'task_index': 2, 'input_time': 2, 'out_time': 1, 'out_channel':1, 'out_size': 1024},
            'tscd':{'task_index': 3, 'input_time': 4, 'out_time': 3, 'out_channel':1, 'out_size': 256},
        }



        self.tss_unify = Dynamic_MLP_OFA_Decoder(out_chans, out_chans)

        # self.tss_unify = nn.Linear(64, 1*2*2)  # for whu
        # self.tss_unify = nn.Linear(32*2, 1*4*4)  # for levircd
        # self.tss_unify = nn.Linear(64, 4*4)  # for whucd
        # self.tss_unify = nn.Linear(32*2, 3*1*1*1)  # for tscd


        ### duplicated init, only affects network weights and has no effect given pretrain
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def decoder_forward(self, x):
        # x = self.ln_pre(x)  # effective for clip model only, otherwise nn.Identity

        for i, blk in enumerate(self.blocks):
            # if (i+1) % (self.interval+1) == 0:
            #     x = rearrange(x, 'b t l c -> b l t c')
            x = blk(x, self.Hp, self.Wp)
            # if (i+1) % (self.interval+1) == 0:
            #     x = rearrange(x, 'b l t c -> b t l c')

        if self.ending_norm:
            # b,t,l,c = x.shape
            # x = rearrange(x, 'b t l c -> (b t) l c')
            x = self.norm(x)  # b h*w c
            # x = rearrange(x, '(b t) l c -> b t l c', b=b, t=t)
        return x
    
    def decoder_in_forward(self, x, dataset_name):
        b,l,c = x.shape
        out_time_length = self.data_task_info[dataset_name]['out_time']

        input_time_length = self.data_task_info[dataset_name]['input_time']
        task_embedding_index = torch.tensor(self.data_task_info[dataset_name]['task_index']).repeat(b).to(device=x.device, dtype=torch.long)
        task_embedding = self.task_embedding(task_embedding_index)  # b,c
        task_embedding = rearrange(task_embedding, 'b (l c) -> b l c', l=1, c=c).repeat(1,l,1)
        x = x + task_embedding
        # print(f'up_forward input_tensor after task_embedding min max:{x.min()}, {x.max()}')
        # print(f'up_forward input_tensor after task_embedding shape:{x.shape}')

        x = self.final_linear(x)  # b,l,c
        # print(f'up_forward input_tensor after final_linear shape:{x.shape}')


        # bt,l,c = x.shape
        # x = rearrange(x, '(b t) l c -> b l (t c)', t=input_time_length)  # 最后一个维度是t个形状为c的tensor排列起来
        # b,l,tc = x.shape
        # # 下面设置的metadata，是c个1到input_time_length的数堆叠起来的数字，与decoder相反,也就是t个形状为c并且值相同的tensor排列起来，是t*c的维度
        # input_time_metadata = torch.cat([torch.full((c,), i) for i in range(1, input_time_length+1)]).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1)  # b,t*c
        # # print(f'input_time_metadata shape:{input_time_metadata.shape}')
        # # print(f'x shape:{x.shape}')
        # x = F.relu(self.temporal_unify_encoder(x, input_time_metadata))  # b,l,c  # ZSJ
        # # x = F.relu(self.temporal_unify_encoder(x))  # b,l,c

        # # 下面设置的metadata，是out_time_length个1到c的数堆叠起来的数字，与encoder相反，也就是c个形状为t并且值相同的tensor排列起来，是c*t的维度
        # # output_time_metadata = torch.cat([torch.full((c,), i) for i in range(1, out_time_length+1)]).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1)  # b,t*c
        # output_time_metadata = torch.cat([torch.full((out_time_length,), i) for i in range(1, c+1)]).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1) / c  # b,t*c

        # # print(f'output_time_metadata shape:{output_time_metadata.shape}')
        # # print(f'x shape:{x.shape}')
        # x = F.relu(self.temporal_unify_linear(x))  # b,l,c
        # # print(f'x after temporal_unify_linear min max:{x.min()}, {x.max()}')
        # x = self.temporal_unify_decoder(x, output_time_metadata)  # b,l,c  # ZSJ
        # # x = self.temporal_unify_decoder(x)  # b,l,c
        # x = rearrange(x, 'b l (t c) -> (b t) l c', t=out_time_length)
        # # print(f'up_forward input_tensor after temporal_unify_decoder min max:{x.min()}, {x.max()}')
        # # print(f'up_forward input_tensor after temporal_unify_decoder shape:{x.shape}')

        bt,l,c = x.shape
        x = rearrange(x, '(b t) l c -> b (l c) t', t=input_time_length)  # 最后一个维度是t个形状为c的tensor排列起来
        b,lc,t = x.shape
        # 下面设置的metadata，是c个1到input_time_length的数堆叠起来的数字，与decoder相反,也就是t个形状为c并且值相同的tensor排列起来，是t*c的维度
        # input_time_metadata = torch.cat([torch.full((c,), i) for i in range(1, input_time_length+1)]).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1)  # b,t*c
        input_time_metadata = torch.arange(1, input_time_length+1).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1)  # b,t
        x = F.relu(self.temporal_unify_encoder(x, input_time_metadata))  # b,l,c  # ZSJ
        # x = self.temporal_unify_encoder(x)  # b,l,c

        # 下面设置的metadata，是out_time_length个1到c的数堆叠起来的数字，与encoder相反，也就是c个形状为t并且值相同的tensor排列起来，是c*t的维度
        # output_time_metadata = torch.cat([torch.full((c,), i) for i in range(1, out_time_length+1)]).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1)  # b,t*c
        # output_time_metadata = torch.cat([torch.full((out_time_length,), i) for i in range(1, c+1)]).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1) / c  # b,t*c
        output_time_metadata = torch.arange(1, out_time_length+1).to(device=x.device, dtype=x.dtype).unsqueeze(0).repeat(b,1)  # b,t

        x = self.temporal_unify_decoder(x, output_time_metadata)  # b,l,c  # ZSJ
        # x = self.temporal_unify_decoder(x)  # b,l,c
        x = rearrange(x, 'b (l c) t -> (b t) l c', l=l)


        return x


    def up_forward(self, x, dataset_name):
        b,l,c = x.shape
        out_time_length = self.data_task_info[dataset_name]['out_time']
        out_channel_length = self.data_task_info[dataset_name]['out_channel']
        out_size_length = self.data_task_info[dataset_name]['out_size']**2

        # 进行统一操作之前，都需要添加task_embedding
        b,l,c = x.shape
        out_time_length = self.data_task_info[dataset_name]['out_time']

        input_time_length = self.data_task_info[dataset_name]['input_time']
        task_embedding_index = torch.tensor(self.data_task_info[dataset_name]['task_index']).repeat(b).to(device=x.device, dtype=torch.long)
        task_embedding = self.task_embedding(task_embedding_index)  # b,c
        task_embedding = rearrange(task_embedding, 'b (l c) -> b l c', l=1, c=c).repeat(1,l,1)
        x = x + task_embedding

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.Hp, w=self.Wp)
        x = self.final(x)  # b,t,l,c
        x = rearrange(x, 'b c h w -> b (h w) c')
        # print(f'up_forward input_tensor after final shape:{x.shape}')

        b,l,c = x.shape
        h, w = self.data_task_info[dataset_name]['out_size'], self.data_task_info[dataset_name]['out_size']
        patch_size = h // int(l ** 0.5)  # 计算 patch 大小
        # tss_metadata = torch.arange(1,patch_size*patch_size+1).repeat(out_channel_length).to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0).repeat(b,out_time_length,1)  # b,t,l
        tss_metadata = torch.arange(1,out_channel_length+1).repeat(patch_size*patch_size).to(device=x.device, dtype=x.dtype).unsqueeze(0).unsqueeze(0).repeat(b,out_time_length,1)  # b,t,l
        tss_metadata = rearrange(tss_metadata, 'b t l -> (b t) l')
        x = self.tss_unify(x, tss_metadata)  # ZSJ
        # x = self.tss_unify(x)

        bt, num_patches, patch_dim = x.shape
        original_c = patch_dim //patch_size//patch_size
        
        # 计算每个维度的patch数量
        h_patches = w_patches = h // patch_size
    
        # 重构patch_dim为(c, patch_size, patch_size)
        x = x.reshape(bt, h_patches, w_patches, original_c, patch_size, patch_size)
    
        # 重新排列维度以准备合并patches
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    
        # 重构为(b*t, c, h, w)
        x = x.reshape(bt, original_c, h, w)

        # x = rearrange(x, '(b t) l c -> b t l c', t=out_time_length)

        # x = rearrange(x, 'b t (h w) (p q c) -> b t (h p) (w q) c', h=h//patch_size, w=w//patch_size, p=patch_size, q=patch_size)

        x = rearrange(x, '(b t) c h w -> b t h w c', t=out_time_length)

        # print(f'up_forward input_tensor after tss_unify min max:{x.min()}, {x.max()}')
        # print(f'up_forward input_tensor after tss_unify shape:{x.shape}')

        return x.contiguous()


    def forward(self, feat, dataset_name, **kwargs):
        # B, C, H,W = feat.shape
        # x = feat.reshape(B, C,-1).permute(0,2,1)
        dataset_name = dataset_name[0]
        feat = self.decoder_in_forward(feat, dataset_name)

        out = self.decoder_forward(feat)

        out = self.up_forward(out, dataset_name)

        return out


def vit_base_patch16_ema(**kwargs):
    backbone = vit_base_patch16(**kwargs)
    backbone.ema = [vit_base_patch16(**kwargs)]
    backbone.ema[0].mask_input = False
    return backbone


class dummy_logger:
    def info(self, **kwargs):
        print(**kwargs)

    def warning(self, **kwargs):
        print(**kwargs)



def load_checkpoint(model, state_dict, load_pos_embed, strict=False, logger=None):
    """
    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    # checkpoint = _load_checkpoint(filename, map_location)
    # OrderedDict is a subclass of dict
    # if not isinstance(checkpoint, dict):
    #     raise RuntimeError(
    #         f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'pos_embed' in state_dict:
        if load_pos_embed:
            state_dict['pos_embed'] = interpolate_pos_embed(pos_embed_checkpoint=state_dict['pos_embed'],
                                                            patch_shape=(model.ori_Hp, model.ori_Hw),
                                                            num_extra_tokens=1)
        else:
            del state_dict['pos_embed']
            print("checkpoint pos_embed removed")

    model_dict = model.state_dict()
    load_dict = {
        k: v for k, v in state_dict.items() if k in model_dict.keys()
    }
    print("Missing keys: {}".format(list(set(model_dict) - set(load_dict))))

    load_state_dict(model, state_dict, strict, logger)


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')


    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)
    print("finish load")


def interpolate_pos_embed(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    embedding_size = pos_embed_checkpoint.shape[-1]
    orig_size = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    # class_token and dist_token are kept unchanged
    print(f"[rank {link.get_rank()}] Position interpolate from {orig_size} to {patch_shape}")
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:] if pos_embed_checkpoint.size(0) == 1 else pos_embed_checkpoint[num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=patch_shape, mode='bicubic', align_corners=False)
    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # (b, h*w, c)
    return new_pos_embed


def interpolate_pos_embed_with_cls_token(pos_embed_checkpoint, patch_shape, num_extra_tokens):
    posemb_tok, posemb_grid = (
        pos_embed_checkpoint[:, :num_extra_tokens],
        pos_embed_checkpoint[0, num_extra_tokens:],
    )
    gs_old_h, gs_old_w = to_2tuple(int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=patch_shape, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, patch_shape[0] * patch_shape[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_size = to_2tuple(grid_size)
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    # import pdb
    # pdb.set_trace()
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_abs_pos(abs_pos, has_cls_token, ori_hw, hw):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.
    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    embed_num, _, emde_dim = abs_pos.size()
    h, w = hw
    if has_cls_token:
     abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))

    ori_hp, ori_hw = ori_hw

    assert ori_hp, ori_hw == xy_num

    if ori_hp != h or ori_hw != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(embed_num, ori_hp, ori_hw, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1).reshape(embed_num, h*w, -1)
    else:
        return abs_pos.reshape(embed_num, h*w, -1)



def Encoder(arch='vit_base', patch_size=(16,16), patch_stride=None, in_chans=227, out_chans=227,
                 pretrained_model=None, finetune_model=None,kwargs=None):

        if patch_stride is None:
            patch_stride =patch_size

        base_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=patch_size, patch_stride=patch_stride, in_chans=in_chans, out_chans=out_chans, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True, 

            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # learnable_pos= True,
        )

        large_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=patch_size,patch_stride=patch_stride,in_chans=in_chans, out_chans=out_chans, embed_dim=1024, depth=24,
            num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # learnable_pos= True,
        )


        huge_default_dict =dict(
            drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
            patch_size=patch_size,patch_stride=patch_stride,in_chans=in_chans, out_chans=out_chans, embed_dim=2048, depth=24,
            num_heads=16, mlp_ratio=4, qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # learnable_pos= True,
        )

        if arch == "vit_base":
            recursive_update(base_default_dict, kwargs)
            encoder = ViT_Encoder(**base_default_dict)

        elif arch == "vit_large":

            recursive_update(large_default_dict, kwargs)
            encoder = ViT_Encoder(**large_default_dict)

        elif arch == "vit_huge":
            recursive_update(huge_default_dict, kwargs)
            encoder = ViT_Encoder(**huge_default_dict)

        else:
            raise Exception("Architecture undefined!")



        if finetune_model is not None:
            import io
            pretrained_dict = torch.load(finetune_model, map_location='cpu')['state_dict']

            model_dict = state_dict()
            pretrained_dict_filter ={}
            for k, v in pretrained_dict.items():
                if k[9:] in model_dict.keys():
                    pretrained_dict_filter.update({k[9:]: v})
            load_state_dict(encoder, pretrained_dict_filter, strict=False, logger=dummy_logger)


            print(
                "Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict_filter)
                                               )))
            model_dict.update(pretrained_dict_filter)
            # import pdb
            # pdb.set_trace()
            load_state_dict(model_dict)
            del pretrained_dict

        return encoder




def Decoder(arch='vit_base', patch_size=(16,16),patch_stride=None, in_chans=227, out_chans=227,
            pretrained_model=None, finetune_model=None, kwargs=None):

    if patch_stride is None:
        patch_stride =patch_size

    base_default_dict =dict(
        drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        patch_size=patch_size, patch_stride=patch_stride, in_chans=in_chans, out_chans=out_chans, embed_dim=768, depth=12,
        num_heads=12, mlp_ratio=4, qkv_bias=True,

        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # learnable_pos= True,
    )

    large_default_dict =dict(
        drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        patch_size=patch_size,patch_stride=patch_stride,in_chans=in_chans, out_chans=out_chans, embed_dim=1024, depth=24,
        num_heads=16, mlp_ratio=4, qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # learnable_pos= True,
    )


    huge_default_dict =dict(
        drop_path_rate=0, use_abs_pos_emb=True,  # as in table 11
        patch_size=patch_size,patch_stride=patch_stride,in_chans=in_chans, out_chans=out_chans, embed_dim=2048, depth=24,
        num_heads=16, mlp_ratio=4, qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # learnable_pos= True,
    )

    if arch == "vit_base":
        recursive_update(base_default_dict, kwargs)
        encoder = ViT_Decoder(**base_default_dict)

    elif arch == "vit_large":

        recursive_update(large_default_dict, kwargs)
        encoder = ViT_Decoder(**large_default_dict)

    elif arch == "vit_huge":
        recursive_update(huge_default_dict, kwargs)
        encoder = ViT_Decoder(**huge_default_dict)

    else:
        raise Exception("Architecture undefined!")



    if finetune_model is not None:
        import io
        pretrained_dict = torch.load(finetune_model, map_location='cpu')['state_dict']

        model_dict = state_dict()
        pretrained_dict_filter ={}
        for k, v in pretrained_dict.items():
            if k[9:] in model_dict.keys():
                pretrained_dict_filter.update({k[9:]: v})
        load_state_dict(encoder, pretrained_dict_filter, strict=False, logger=dummy_logger)


        print(
            "Missing keys: {}".format(list(set(model_dict) - set(pretrained_dict_filter)
                                           )))
        model_dict.update(pretrained_dict_filter)
        # import pdb
        # pdb.set_trace()
        load_state_dict(model_dict)
        del pretrained_dict

    return encoder





