"""
mlp_mixer.py — MLP-Mixer for vision.

Reference
---------
Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision",
NeurIPS 2021.

Overview
--------
MLP-Mixer replaces both self-attention *and* convolutional token mixing
with two simple MLPs:

  * Token-mixing MLP  : operates on the patch (token) axis, shared
                        across channels.
  * Channel-mixing MLP: operates on the channel axis, shared across
                        patches (this is the standard ViT MLP).

There is no inductive bias from attention, convolution, or frequency
transforms.  In small-data regimes this is a known weakness, which makes
Mixer-Tiny an informative baseline: it quantifies how much the PSPN /
GFNet / AFNO spectral inductive bias buys you on 91K canine images.

This file provides:
  * MixerBlock        — token-mix MLP + channel-mix MLP with residuals
  * MLPMixer          — full backbone (patch embed + L blocks + head)
  * mlp_mixer_tiny    — variant param-matched to PSPN-GFNet-Ti / GFNet-tiny
                        at 224×224 (dim=192, depth=12, 14×14 token grid)
  * mlp_mixer_small, mlp_mixer_base — larger variants
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


# ======================================================================
#  Basic pieces
# ======================================================================
class Mlp(nn.Module):
    """Standard 2-layer MLP with GELU, used for both token and channel
    mixing."""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x.div(keep) * mask


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size // patch_size
        self.grid_w = img_size // patch_size
        self.num_tokens = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                            # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)            # (B, N, D)
        return x

class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, dim: int,
                 tokens_mlp_ratio: float = 0.5,
                 channels_mlp_ratio: float = 4.0,
                 drop: float = 0., drop_path: float = 0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        token_hidden = max(1, int(num_tokens * tokens_mlp_ratio))
        channel_hidden = max(1, int(dim * channels_mlp_ratio))

        self.norm1 = norm_layer(dim)

        self.token_mix = Mlp(num_tokens, token_hidden,
                              num_tokens, act_layer=act_layer, drop=drop)

        self.norm2 = norm_layer(dim)
        self.channel_mix = Mlp(dim, channel_hidden,
                                dim, act_layer=act_layer, drop=drop)

        self.drop_path = (DropPath(drop_path)
                          if drop_path > 0 else nn.Identity())

    def forward(self, x):
        # x : (B, N, D)
        y = self.norm1(x)
        y = y.transpose(1, 2)                       # (B, D, N)
        y = self.token_mix(y)                       # (B, D, N)
        y = y.transpose(1, 2)                       # (B, N, D)
        x = x + self.drop_path(y)

        x = x + self.drop_path(self.channel_mix(self.norm2(x)))
        return x

class MLPMixer(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=11,
                 embed_dim=192, depth=12,
                 tokens_mlp_ratio=0.5, channels_mlp_ratio=4.0,
                 drop_rate=0., drop_path_rate=0.1,
                 representation_size: Optional[int] = None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                       in_chans=in_chans, embed_dim=embed_dim)
        num_tokens = self.patch_embed.num_tokens
        self.num_tokens = num_tokens
                   
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            MixerBlock(num_tokens=num_tokens, dim=embed_dim,
                        tokens_mlp_ratio=tokens_mlp_ratio,
                        channels_mlp_ratio=channels_mlp_ratio,
                        drop=drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size), nn.Tanh()
            )
            head_in = representation_size
        else:
            self.pre_logits = nn.Identity()
            head_in = embed_dim
        self.head = (nn.Linear(head_in, num_classes)
                     if num_classes > 0 else nn.Identity())

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.patch_embed(x)                     # (B, N, D)
        x = self.pos_drop(x)
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = cp.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        x = self.norm(x)
        return x.mean(dim=1)                        # (B, D)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pre_logits(x)
        return self.head(x)

def mlp_mixer_tiny(num_classes=11, **kw):
    """Param-matched to PSPN-GFNet-Ti / GFNet-tiny at 224×224.

    Expected params: ~4.2M (dim=192, depth=12, 14×14 grid → 196 tokens,
    token-mix hidden 98, channel-mix hidden 768).
    """
    return MLPMixer(embed_dim=192, depth=12,
                     tokens_mlp_ratio=0.5,
                     channels_mlp_ratio=4.0,
                     num_classes=num_classes, **kw)


def mlp_mixer_small(num_classes=11, **kw):
    """Mixer-S/16 style (dim=512, depth=8 per paper; we use 12 here to
    match the depth of other tiny baselines)."""
    return MLPMixer(embed_dim=384, depth=12,
                     tokens_mlp_ratio=0.5,
                     channels_mlp_ratio=4.0,
                     num_classes=num_classes, **kw)


def mlp_mixer_base(num_classes=11, **kw):
    """Mixer-B/16 style."""
    return MLPMixer(embed_dim=768, depth=12,
                     tokens_mlp_ratio=0.5,
                     channels_mlp_ratio=4.0,
                     num_classes=num_classes, **kw)


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    token_mix = sum(p.numel() for n, p in model.named_parameters()
                     if 'token_mix' in n)
    channel_mix = sum(p.numel() for n, p in model.named_parameters()
                       if 'channel_mix' in n)
    return {'total': total, 'trainable': trainable,
            'token_mix': token_mix, 'channel_mix': channel_mix}

if __name__ == '__main__':
    torch.manual_seed(0)

    print('=' * 64)
    print(' MLPMixer variants — sizes (224 input, patch=16)')
    print('=' * 64)
    for fn, name in [(mlp_mixer_tiny, 'tiny'),
                      (mlp_mixer_small, 'small'),
                      (mlp_mixer_base, 'base')]:
        m = fn(num_classes=11)
        pc = count_params(m)
        print(f'  {name:5s}  dim={m.embed_dim:3d}  tokens={m.num_tokens:3d}  '
              f"total={pc['total']:>11,}  "
              f"token_mix={pc['token_mix']:>8,}  "
              f"channel_mix={pc['channel_mix']:>9,}")

    print()
    print('=' * 64)
    print(' Mixer-Tiny forward/backward sanity')
    print('=' * 64)
    m = mlp_mixer_tiny(num_classes=11)
    m.train()
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print(f'  forward : logits={tuple(y.shape)}')
    loss = y.sum(); loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2
                    for p in m.parameters() if p.grad is not None) ** 0.5
    print(f'  backward: grad_norm={grad_norm:.3f}')
    print('All checks passed.')
