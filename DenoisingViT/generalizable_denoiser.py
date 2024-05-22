from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block, Mlp
from torch import Tensor

from DenoisingViT.vit_wrapper import ViTWrapper


class Denoiser(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 37,
        noise_map_width: int = 37,
        feature_dim: int = 768,
        vit: ViTWrapper = None,
        enable_pe: bool = False,
        denoiser_type: str = "transformer",
        finetune_vit: bool = False,
        finetune_pe: bool = False,
        with_layer_norm: bool = True,
    ):
        super().__init__()
        # default values
        self.denoiser_type = denoiser_type
        self.pos_embed = None
        self.denoiser = None
        self.with_layer_norm = with_layer_norm
        self.vit = vit
        self.finetune_vit = finetune_vit
        self.finetune_pe = finetune_pe
        self.enable_vit_grad = finetune_vit or finetune_pe
        if not self.enable_vit_grad:
            if self.denoiser_type == "transformer":
                self.denoiser = Block(
                    dim=feature_dim,
                    num_heads=feature_dim // 64,
                    mlp_ratio=4,
                    qkv_bias=True,
                    qk_norm=False,
                    init_values=None,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    mlp_layer=Mlp,
                )
                if self.with_layer_norm:
                    print("Using layer norm")
                    self.denoiser = nn.Sequential(
                        self.denoiser,
                        nn.LayerNorm(feature_dim, eps=1e-6),
                    )
            elif self.denoiser_type == "conv1x1":
                self.denoiser = nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim // 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim // 2, feature_dim // 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim // 2, feature_dim, 1),
                )
            elif self.denoiser_type == "conv3x3":
                self.denoiser = nn.Sequential(
                    nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim // 2, feature_dim // 2, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(feature_dim // 2, feature_dim, 3, padding=1),
                )

            self.enable_pe = enable_pe
            if self.enable_pe:
                self.pos_embed = nn.Parameter(
                    torch.randn(
                        1,
                        noise_map_height * noise_map_width,
                        feature_dim,
                    )
                    * 0.02,
                    requires_grad=True,
                )
        if self.vit is not None:
            for param in self.vit.parameters():
                param.requires_grad = self.finetune_vit
            if self.finetune_pe:
                self.vit.pos_embed.requires_grad = True

    def forward_features(self, x: Tensor) -> Tensor:
        outputs = self.forward_features_with_cls_tokens(x)
        patches = outputs[0].reshape(outputs[0].size(0), -1, outputs[0].size(-1))
        cls_token = outputs[1]
        cls_token_denoised = outputs[2]
        return patches, cls_token, cls_token_denoised

    def forward_features_with_cls_tokens(self, x: Tensor, norm=True) -> Tensor:
        # run backbone if backbone is there
        raw_vit_feats = None
        if self.vit is not None:
            with torch.set_grad_enabled(self.enable_vit_grad):
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[self.vit.last_layer_index],
                    reshape=True,
                    return_class_token=True,
                    norm=norm,
                )
                vit_outputs = vit_outputs[-1]
                raw_vit_feats = vit_outputs[0].permute(0, 2, 3, 1)
                if not self.enable_vit_grad:
                    raw_vit_feats = raw_vit_feats.detach()
                x = raw_vit_feats
                cls_token = vit_outputs[1]
        B, H, W, C = x.shape
        if not self.enable_vit_grad:
            x = x.reshape(B, H * W, C)
            if self.enable_pe:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed, (H, W), num_prefix_tokens=0
                )
                x = x + pos_embed.repeat(B, 1, 1)
            x_with_cls_token = torch.cat([cls_token.unsqueeze(1), x], dim=1)
            x_with_cls_token = self.denoiser(x_with_cls_token)
            x = x_with_cls_token[:, 1:]
            cls_token_denoised = x_with_cls_token[:, 0]
            out_feat = x.reshape(B, H, W, C)

            raw_vit_feats = (
                raw_vit_feats.reshape(B, H, W, C) if raw_vit_feats is not None else None
            )
            return out_feat, cls_token, cls_token_denoised
        else:
            out_feat = raw_vit_feats
            raw_vit_feats = (
                raw_vit_feats.reshape(B, H, W, C) if raw_vit_feats is not None else None
            )
            return out_feat, cls_token, cls_token_denoised

    def forward(
        self,
        x: Tensor,
        return_prefix_tokens=False,
        return_class_token=False,
        norm=True,
        return_dict=False,
        return_channel_first=False,
    ) -> Tensor:
        # run backbone if backbone is there
        prefix_tokens, raw_vit_feats = None, None
        if self.vit is not None:
            with torch.set_grad_enabled(self.enable_vit_grad):
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[self.vit.last_layer_index],
                    reshape=True,
                    return_prefix_tokens=return_prefix_tokens,
                    return_class_token=return_class_token,
                    norm=norm,
                )
                vit_outputs = (
                    vit_outputs[-1]
                    if return_prefix_tokens or return_class_token
                    else vit_outputs
                )
                raw_vit_feats = vit_outputs[0].permute(0, 2, 3, 1)
                if not self.enable_vit_grad:
                    raw_vit_feats = raw_vit_feats.detach()
                x = raw_vit_feats
                if return_prefix_tokens or return_class_token:
                    prefix_tokens = vit_outputs[1]
        B, H, W, C = x.shape
        if not self.enable_vit_grad:
            if self.denoiser_type == "transformer":
                x = x.reshape(B, H * W, C)
                if self.enable_pe:
                    pos_embed = resample_abs_pos_embed(
                        self.pos_embed, (H, W), num_prefix_tokens=0
                    )
                    x = x + pos_embed.repeat(B, 1, 1)
                x = self.denoiser(x)
                out_feat = x.reshape(B, H, W, C)
            else:
                out_feat = self.denoiser(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            raw_vit_feats = (
                raw_vit_feats.reshape(B, H, W, C) if raw_vit_feats is not None else None
            )
            if return_channel_first:
                out_feat = out_feat.permute(0, 3, 1, 2)
                raw_vit_feats = (
                    raw_vit_feats.permute(0, 3, 1, 2)
                    if raw_vit_feats is not None
                    else None
                )
            if return_dict:
                return {
                    "pred_denoised_feats": out_feat,
                    "raw_vit_feats": raw_vit_feats,
                    "prefix_tokens": prefix_tokens,
                }
            if prefix_tokens is not None:
                return out_feat, prefix_tokens
            return out_feat
        else:
            out_feat = raw_vit_feats
            raw_vit_feats = (
                raw_vit_feats.reshape(B, H, W, C) if raw_vit_feats is not None else None
            )
            if return_channel_first:
                out_feat = out_feat.permute(0, 3, 1, 2)
                raw_vit_feats = (
                    raw_vit_feats.permute(0, 3, 1, 2)
                    if raw_vit_feats is not None
                    else None
                )
            if return_dict:
                return {
                    "pred_denoised_feats": out_feat,
                    "raw_vit_feats": raw_vit_feats,
                    "prefix_tokens": prefix_tokens,
                }
            if prefix_tokens is not None:
                return out_feat, prefix_tokens
            return out_feat
