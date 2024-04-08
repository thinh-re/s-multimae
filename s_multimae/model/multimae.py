import math
import re
from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torchvision.ops import MLP
from einops import rearrange, repeat
from torch import Tensor, nn

from definition import PRETRAINED_BACKBONE
from ..configs.base_config import base_cfg
from ..utils import count_parameters
from .components import (
    build_2d_sincos_posemb,
    drop_path,
    pair,
    trunc_normal_,
)


class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(
        self,
        num_channels: int,
        stride_level: int,
        patch_size_full: Union[int, Tuple[int, int]],
        dim_tokens: Optional[int] = None,
        sincos_pos_emb: bool = True,
        learnable_pos_emb: bool = False,
        image_size: Union[int, Tuple[int]] = 224,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = pair(patch_size_full)
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = pair(image_size)
        self.num_patches = (self.image_size[0] // patch_size_full) * (
            self.image_size[1] // patch_size_full
        )

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size_full[0] // stride_level)
        self.P_W = max(1, self.patch_size_full[1] // stride_level)

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        h_posemb = self.image_size[0] // (self.stride_level * self.P_H)
        w_posemb = self.image_size[1] // (self.stride_level * self.P_W)
        if self.sincos_pos_emb:
            self.pos_emb = build_2d_sincos_posemb(
                h=h_posemb, w=w_posemb, embed_dim=self.dim_tokens
            )
            self.pos_emb = nn.Parameter(
                self.pos_emb, requires_grad=self.learnable_pos_emb
            )
        else:
            self.pos_emb = nn.Parameter(
                torch.zeros(1, self.dim_tokens, h_posemb, w_posemb)
            )
            trunc_normal_(self.pos_emb, std=0.02)

        # Image -> tokens projection
        self.proj = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.dim_tokens,
            kernel_size=(self.P_H, self.P_W),
            stride=(self.P_H, self.P_W),
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_emb"}

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input image tensor
        """
        B, C, H, W = x.shape
        assert (
            self.dim_tokens is not None
        ), "Need to call init(dim_tokens) function first"
        assert (H % self.P_H == 0) and (
            W % self.P_W == 0
        ), f"Image sizes {H}x{W} must be divisible by patch sizes {self.P_H}x{self.P_W}"
        N_H, N_W = H // self.P_H, W // self.P_W  # Number of patches in height and width

        # Create patches [B, C, H, W] -> [B, (H*W), C]
        projected_x = self.proj(x)
        x_patch = rearrange(projected_x, "b d nh nw -> b (nh nw) d")

        # Create positional embedding
        x_pos_emb = F.interpolate(
            self.pos_emb, size=(N_H, N_W), mode="bicubic", align_corners=False
        )
        x_pos_emb = rearrange(x_pos_emb, "b d nh nw -> b (nh nw) d")

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path: Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 0 (disabled for isotropic ConvNeXt).

    Code from: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtAdapter(nn.Module):
    """Output adapter with ConvNext blocks for semantic segmentation

    :param num_classes: Number of classes
    :param num_heads: Number of attention heads
    :param embed_dim: Token dimension after projection, and before reshaping operation.
    :param preds_per_patch: Increases size of feature map by reshaping each patch  Each patch gets reshaped
        from embed_dim x 1 x 1 to (embed_dim / preds_per_patch) x (preds_per_patch ** 0.5) x (preds_per_patch ** 0.5)
    :param main_tasks: Tasks to use for the adapter. Only tokens coming from these tasks are kept.
    :param patch_size: Size of patches
    :param depth: Number of ConvNeXt blocks
    :interpolate_mode: Interpolation mode for final upsampling
    """

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        embed_dim: int = 6144,
        preds_per_patch: int = 16,
        main_tasks: Iterable[str] = ("rgb",),
        patch_size: int = 16,
        depth: int = 4,
        interpolate_mode: str = "bilinear",
        act_fn: nn.Module = nn.GELU,
        dec_kernel: int = 1,
    ):
        super().__init__()
        self.main_tasks = main_tasks
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.preds_per_patch = preds_per_patch
        self.class_dim = embed_dim // preds_per_patch
        self.num_classes = num_classes
        self.interpolate_mode = interpolate_mode
        self.image_size = image_size

        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=self.class_dim) for _ in range(depth)]
        )
        if dec_kernel == 1:
            self.final_layer_1 = nn.Sequential(
                nn.Conv2d(self.class_dim, self.class_dim // 4, 1),
                nn.BatchNorm2d(self.class_dim // 4),
                act_fn(),
                nn.Upsample(scale_factor=2, mode=self.interpolate_mode),
            )

            self.final_layer_2 = nn.Sequential(
                nn.Conv2d(self.class_dim // 4, self.class_dim // 16, 1),
                nn.BatchNorm2d(self.class_dim // 16),
                act_fn(),
                nn.Upsample(size=image_size, mode=self.interpolate_mode),
            )

            self.final_layer = nn.Conv2d(self.class_dim // 16, self.num_classes, 1)
        elif dec_kernel == 3:
            self.final_layer_1 = nn.Sequential(
                nn.Conv2d(
                    self.class_dim,
                    self.class_dim // 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(self.class_dim // 4),
                act_fn(),
                nn.Upsample(scale_factor=2, mode=self.interpolate_mode),
            )

            self.final_layer_2 = nn.Sequential(
                nn.Conv2d(
                    self.class_dim // 4,
                    self.class_dim // 16,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(self.class_dim // 16),
                act_fn(),
                nn.Upsample(size=image_size, mode=self.interpolate_mode),
            )

            self.final_layer = nn.Conv2d(
                self.class_dim // 16,
                self.num_classes,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            raise Exception(f"Unsupported dec_kernel {dec_kernel}")

        self.apply(self._init_weights)

    def init(self, dim_tokens_enc: int = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.in_channels = dim_tokens_enc * len(self.main_tasks)

        # Projection of encoder tokens to the patch dimension
        self.proj_dec = nn.Linear(self.in_channels, self.embed_dim)
        self._init_weights(self.proj_dec)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def adapt_tokens(self, encoder_tokens: Tensor, input_info: Dict):
        # Adapt tokens
        x = []
        for task in self.main_tasks:
            start_idx = input_info["tasks"][task]["start_idx"]
            end_idx = input_info["tasks"][task]["end_idx"]
            x.append(encoder_tokens[:, start_idx:end_idx])

        x = torch.cat(x, dim=-1)
        return x

    def forward(self, encoder_tokens: Tensor, input_info: Dict) -> Tensor:
        H, W = input_info["image_size"]
        N_H, N_W = H // self.patch_size, W // self.patch_size

        x = self.adapt_tokens(encoder_tokens, input_info)

        x = self.proj_dec(x)
        x = rearrange(
            x,
            "b n (p c) -> b (n p) c",
            n=N_H * N_W,
            p=self.preds_per_patch,
            c=self.class_dim,
        )
        x = rearrange(
            x,
            "b (nh nw ph pw) c -> b c (nh ph) (nw pw)",
            nh=N_H,
            nw=N_W,
            ph=int(self.preds_per_patch**0.5),
            pw=int(self.preds_per_patch**0.5),
        )

        x = self.blocks(x)

        # for block in self.blocks:
        #     x = block(x)
        #     print(x.shape)

        # print(x.shape)
        x = self.final_layer_1(x)
        # print(x.shape)
        x = self.final_layer_2(x)
        # print(x.shape)
        x = self.final_layer(x)
        # print(x.shape)

        # Interpolate to sod res
        # x = F.interpolate(x, size=(H, W), mode=self.interpolate_mode)

        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MultiMAE(nn.Module):
    """MultiMAE: Multi-task Multi-modal Masked Autoencoder
    This module performs masking in its forward pass.
    The MultiViT module defined below inherits from this module and performs a regular forward pass,
    and should be used instead for downstream tasks


    :param input_adapters: Dictionary of task -> input adapters
    :param output_adapters: Optional dictionary of task -> output adapters

    :param num_global_tokens: Number of additional global tokens to add (like cls tokens), default is 1
    :param dim_tokens: Dimension of encoder tokens
    :param depth: Depth of encoder
    :param num_heads: Number of attention heads
    :param mlp_ratio: MLP hidden dim ratio
    :param qkv_bias: Set to False to disable bias
    :param drop_rate: Dropout after MLPs and Attention
    :param attn_drop_rate: Attention matrix drop rate
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """

    def __init__(
        self,
        input_adapters: Dict[str, PatchedInputAdapter],
        output_adapters: Dict[str, ConvNeXtAdapter],
        num_global_tokens: int = 1,
        dim_tokens: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        freeze_encoder: bool = False,
        num_additional_gt_tokens: int = 0,  # @deprecated
        actual_num_additional_gt_tokens: int = 0,  # @deprecated
        learnable_additional_gt_tokens: bool = False,
        additional_gt_tokens_mlp_channels: List[int] = [],
        ground_truth_version: int = -1,
        A: float = 0.5,
    ):
        super().__init__()
        self.dim_tokens = dim_tokens
        self.ground_truth_version = ground_truth_version
        # Initialize input and output adapters
        for adapter in input_adapters.values():
            adapter.init(dim_tokens=dim_tokens)
        self.input_adapters = nn.ModuleDict(input_adapters)
        for adapter in output_adapters.values():
            adapter.init(dim_tokens_enc=dim_tokens)
        self.output_adapters = nn.ModuleDict(output_adapters)

        # Additional learnable tokens that can be used by encoder to process/store global information
        self.num_global_tokens = num_global_tokens
        self.global_tokens = nn.Parameter(torch.zeros(1, num_global_tokens, dim_tokens))
        trunc_normal_(self.global_tokens, std=0.02)

        self.num_additional_gt_tokens = num_additional_gt_tokens  # @deprecated
        self.actual_num_additional_gt_tokens = (
            actual_num_additional_gt_tokens  # @deprecated
        )
        self.A = A
        self.additional_gt_tokens_mlp_channels = additional_gt_tokens_mlp_channels
        self.learnable_additional_gt_tokens = learnable_additional_gt_tokens
        self.init_gt_tokens()

        # Transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.encoder = nn.Sequential(
            *[
                Block(
                    dim=dim_tokens,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        print(f"Encoder {count_parameters(self.encoder)}")

        if freeze_encoder:
            print("Freeze encoder")
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.apply(self._init_weights)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                elif "kv" in name:
                    # treat the weights of K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 2 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)

            if isinstance(m, nn.Conv2d):
                if ".proj" in name:
                    # From MAE, initialize projection like nn.Linear (instead of nn.Conv2d)
                    w = m.weight.data
                    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        print(f"Total params: {count_parameters(self)}")

    def init_gt_tokens(self):
        """Just prepare beforehand to save time in training
        In inference, there is no need"""
        addtional_gt_tokens: List[Tensor] = []
        if self.num_additional_gt_tokens == 0:
            self.token_mlp = nn.Identity()
            return
        if len(self.additional_gt_tokens_mlp_channels) > 0:
            self.token_mlp = MLP(
                self.dim_tokens,
                self.additional_gt_tokens_mlp_channels + [self.dim_tokens],
            )
        else:
            self.token_mlp = nn.Identity()

        if self.ground_truth_version != 6:
            T = 1 / (self.num_additional_gt_tokens * 4)
            for i in range(self.actual_num_additional_gt_tokens):
                t = [
                    2 * math.pi * (offset / self.dim_tokens - i * T)
                    for offset in range(self.dim_tokens)
                ]
                addtional_gt_tokens.append(
                    nn.Parameter(
                        self.A * torch.cos(Tensor(t).unsqueeze(0).unsqueeze(0)),
                        requires_grad=self.learnable_additional_gt_tokens,
                    )
                )
            self.addtional_gt_tokens = nn.ParameterList(addtional_gt_tokens)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_set = {"global_tokens"}

        for task, adapter in self.input_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f"input_adapters.{task}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        for task, adapter in self.output_adapters.items():
            if hasattr(adapter, "no_weight_decay"):
                to_skip = adapter.no_weight_decay()
                to_skip = set([f"output_adapters.{task}.{name}" for name in to_skip])
                no_wd_set = no_wd_set | to_skip

        return no_wd_set

    def generate_input_info(
        self, input_task_tokens: Dict[str, Tensor], image_size: Tuple[int, int]
    ) -> Dict[str, Tensor]:
        input_info = OrderedDict()
        i = 0
        input_info["tasks"] = {}
        for domain, tensor in input_task_tokens.items():
            num_tokens: Union[int, Tensor] = tensor.shape[1]

            if type(num_tokens) == Tensor:
                num_tokens = num_tokens.item()

            d = {
                "num_tokens": num_tokens,
                "has_2d_posemb": True,
                "start_idx": i,
                "end_idx": i + num_tokens,
            }
            i += num_tokens
            input_info["tasks"][domain] = d

        input_info["image_size"] = image_size
        input_info["num_task_tokens"] = i
        input_info["num_global_tokens"] = self.num_global_tokens

        return input_info


class MultiViT(MultiMAE):
    def extract_B_H_W(self, x: Dict[str, Tensor]) -> Tuple[int, int, int]:
        # If input x is a Tensor, assume it's RGB
        # x = {'rgb': x} if isinstance(x, Tensor) else x
        # Need image size for tokens->image reconstruction
        if "rgb" in x:
            B, _, H, W = x["rgb"].shape
        elif "sod" in x:
            B, H, W = x["sod"].shape
            H *= self.input_adapters["sod"].stride_level
            W *= self.input_adapters["sod"].stride_level
        else:
            B, _, H, W = list(x.values())[0].shape
        return B, H, W

    def process_input(
        self,
        x: Dict[str, Tensor],
        gt_index_lst: List[int],
        num_gts_lst: List[int],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        len(gt_i) must equal to x.shape[0] when self.num_additional_gt_tokens > 0
        """
        B, H, W = self.extract_B_H_W(x)

        # Encode selected inputs to tokens
        input_task_tokens: Dict[str, Tensor] = {
            domain: self.input_adapters[domain](tensor)
            for domain, tensor in x.items()
            if domain in self.input_adapters
        }

        input_info = self.generate_input_info(
            input_task_tokens=input_task_tokens, image_size=(H, W)
        )
        input_tokens = torch.cat(
            [task_tokens for task_tokens in input_task_tokens.values()], dim=1
        )

        # Add global tokens to input tokens
        global_tokens = repeat(self.global_tokens, "() n d -> b n d", b=B)

        if self.ground_truth_version == 6:
            # We need two inputs: gt_index, num_gts
            assert len(gt_index_lst) == len(num_gts_lst)
            additional_gt_tokens = []
            for gt_index, num_gts in zip(gt_index_lst, num_gts_lst):
                T = 1 / num_gts
                i = gt_index
                t = [
                    2 * math.pi * (offset / self.dim_tokens - i * T)
                    for offset in range(self.dim_tokens)
                ]
                additional_gt_token = self.A * torch.cos(
                    Tensor(t).unsqueeze(0).unsqueeze(0)
                )
                additional_gt_tokens.append(additional_gt_token)
            additional_gt_tokens = torch.cat(additional_gt_tokens, dim=0).to(
                input_tokens.device
            )
            additional_gt_tokens = self.token_mlp(additional_gt_tokens)
            input_tokens = torch.cat(
                [input_tokens, global_tokens, additional_gt_tokens], dim=1
            )
        else:
            if self.num_additional_gt_tokens > 0:

                assert gt_index_lst is not None and len(gt_index_lst) == B
                additional_gt_tokens: Tensor = torch.cat(
                    [self.addtional_gt_tokens[gt_i] for gt_i in gt_index_lst], dim=0
                )
                additional_gt_tokens = self.token_mlp(additional_gt_tokens)
                input_tokens = torch.cat(
                    [input_tokens, global_tokens, additional_gt_tokens], dim=1
                )
            else:
                input_tokens = torch.cat([input_tokens, global_tokens], dim=1)

        return input_tokens, input_info

    def forward(
        self,
        x: Dict[str, Tensor],
        gt_index_lst: Optional[List[int]] = None,
        max_gts_lst: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass through input adapters, transformer encoder and output adapters.

        :param x: Dictionary of tensors
        :param outputs: List of outputs. For ex: outputs=['sod', 'depth']. Make sure 'sod' placed first!
        """
        input_tokens, input_info = self.process_input(x, gt_index_lst, max_gts_lst)

        # Pass tokens through Transformer
        encoder_tokens = self.encoder(input_tokens)

        # Decode tokens for each task using task-specific output adapters
        preds = {
            domain: self.output_adapters[domain](
                encoder_tokens=encoder_tokens,
                input_info=input_info,
            )
            for domain in self.output_adapters
        }

        return preds


def interpolate_pos_embed_multimae(
    model: MultiViT,
    checkpoint_model: Dict[str, Tensor],
) -> None:
    pattern = "input_adapters\.(.*)\.pos_emb"
    matched_keys = [k for k in checkpoint_model if bool(re.match(pattern, k))]

    for key in matched_keys:
        domain = re.match(pattern, key).group(1)  # group(0) is entire matched regex
        if getattr(model.input_adapters, domain, None) is not None:
            pos_embed_checkpoint = checkpoint_model[key]
            _, _, orig_H, orig_W = pos_embed_checkpoint.shape
            _, _, new_H, new_W = getattr(model.input_adapters, domain).pos_emb.shape
            if (orig_H != new_H) or (orig_W != new_W):
                print(
                    f"Key {key}: Position interpolate from {orig_H}x{orig_W} to {new_H}x{new_W}"
                )
                pos_embed_checkpoint = torch.nn.functional.interpolate(
                    pos_embed_checkpoint,
                    size=(new_H, new_W),
                    mode="bicubic",
                    align_corners=False,
                )
                checkpoint_model[key] = pos_embed_checkpoint


def construct_adapters(cfg: base_cfg):
    INPUT_ADAPTERS = {
        "rgb": PatchedInputAdapter(
            num_channels=3,
            stride_level=1,
            patch_size_full=cfg.input_patch_size,
            image_size=cfg.image_size,
            learnable_pos_emb=cfg.learnable_pos_emb,
        ),
        "depth": PatchedInputAdapter(
            num_channels=1,
            stride_level=1,
            patch_size_full=cfg.input_patch_size,
            image_size=cfg.image_size,
            learnable_pos_emb=cfg.learnable_pos_emb,
        ),
    }

    num_classes = cfg.num_classes
    if cfg.ground_truth_version in [5, 6]:
        num_classes = 1

    OUTPUT_ADAPTERS = {
        "sod": partial(
            ConvNeXtAdapter,
            num_classes=num_classes,
            image_size=cfg.image_size,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.input_patch_size,
            preds_per_patch=cfg.output_patch_size,
            depth=cfg.decoder_depth,
            interpolate_mode=cfg.decoder_interpolate_mode,
            main_tasks=cfg.decoder_main_tasks,
            act_fn=cfg.act_fn,
            dec_kernel=cfg.dec_kernel,
        ),
        "rgb": partial(
            ConvNeXtAdapter,
            num_classes=3,
            image_size=cfg.image_size,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.input_patch_size,
            preds_per_patch=cfg.output_patch_size,
            depth=cfg.decoder_depth,
            interpolate_mode=cfg.decoder_interpolate_mode,
            main_tasks=cfg.decoder_main_tasks,
            act_fn=cfg.act_fn,
            dec_kernel=cfg.dec_kernel,
        ),
        "depth": partial(
            ConvNeXtAdapter,
            num_classes=1,
            image_size=cfg.image_size,
            embed_dim=cfg.embed_dim,
            patch_size=cfg.input_patch_size,
            preds_per_patch=cfg.output_patch_size,
            depth=cfg.decoder_depth,
            interpolate_mode=cfg.decoder_interpolate_mode,
            main_tasks=cfg.decoder_main_tasks,
            act_fn=cfg.act_fn,
            dec_kernel=cfg.dec_kernel,
        ),
    }

    if cfg.ground_truth_version == 3:
        for i in range(cfg.num_classes):
            OUTPUT_ADAPTERS[f"sod{i}"] = partial(
                ConvNeXtAdapter,
                num_classes=1,
                image_size=cfg.image_size,
                embed_dim=cfg.embed_dim,
                patch_size=cfg.input_patch_size,
                preds_per_patch=cfg.output_patch_size,
                depth=cfg.decoder_depth,
                interpolate_mode=cfg.decoder_interpolate_mode,
                main_tasks=cfg.decoder_main_tasks,
                act_fn=cfg.act_fn,
                dec_kernel=cfg.dec_kernel,
            )
    return INPUT_ADAPTERS, OUTPUT_ADAPTERS


def generate_smultimae_model(cfg: base_cfg) -> Tuple[MultiViT, List[Dict]]:
    """MULTIMAE"""
    assert len(cfg.decoder_main_tasks) == len(
        cfg.outputs
    ), "Length of decoder main tasks must match length of outputs"

    INPUT_ADAPTERS, OUTPUT_ADAPTERS = construct_adapters(cfg)

    input_adapters = dict()
    for input_key in cfg.inputs:
        input_adapters[input_key] = INPUT_ADAPTERS[input_key]

    output_adapters = dict()
    for output_key, decoder_main_tasks_per_output in zip(
        cfg.outputs, cfg.decoder_main_tasks
    ):
        output_adapters[output_key] = OUTPUT_ADAPTERS[output_key](
            main_tasks=decoder_main_tasks_per_output
        )

    num_additional_gt_tokens = 0  # @deprecated
    actual_num_additional_gt_tokens = 0  # @deprecated
    if cfg.ground_truth_version in [5, 6]:  # @deprecated
        num_additional_gt_tokens = cfg.num_classes  # @deprecated
        actual_num_additional_gt_tokens = cfg.actual_num_classes  # @deprecated
    model = MultiViT(
        input_adapters=input_adapters,
        output_adapters=output_adapters,
        freeze_encoder=cfg.freeze_encoder,
        drop_path_rate=0.1,
        dim_tokens=cfg.dim_tokens,
        depth=cfg.encoder_depth,
        num_heads=cfg.num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_additional_gt_tokens=num_additional_gt_tokens,  # @deprecated
        actual_num_additional_gt_tokens=actual_num_additional_gt_tokens,  # @deprecated
        ground_truth_version=cfg.ground_truth_version,
    )

    # return load_pretrained_backbone(cfg, model)
    return model, []
