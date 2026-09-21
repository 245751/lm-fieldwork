"""Run each attention step on tiny tensors and optionally print shapes."""

from __future__ import annotations

import argparse

import torch

from lm_fieldwork.attention import ToySelfAttention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-shapes", action="store_true")
    parser.add_argument("--no-causal-mask", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(7)
    batch, time, channels, heads = 2, 4, 8, 2
    x = torch.randn(batch, time, channels)
    layer = ToySelfAttention(channels, heads)
    output, weights = layer(x, causal=not args.no_causal_mask)

    if args.show_shapes:
        q, k, v = layer.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, time, heads, layer.head_dim).transpose(1, 2)

        q_heads, k_heads, v_heads = split_heads(q), split_heads(k), split_heads(v)
        scores = q_heads @ k_heads.transpose(-2, -1)
        attended = weights @ v_heads
        merged = attended.transpose(1, 2).contiguous().view(batch, time, channels)
        print(f"input:            {tuple(x.shape)} = (B, T, C)")
        print(f"Q/K/V projected:  {tuple(q.shape)} = (B, T, C)")
        print(f"Q/K/V per head:   {tuple(q_heads.shape)} = (B, H, T, D)")
        print(f"scores Q @ K^T:   {tuple(scores.shape)} = (B, H, T, T)")
        print(f"weights:          {tuple(weights.shape)} = (B, H, T, T)")
        print(f"weights @ V:      {tuple(attended.shape)} = (B, H, T, D)")
        print(f"merged heads:     {tuple(merged.shape)} = (B, T, C)")
        print(f"output projection: {tuple(output.shape)} = (B, T, C)")
    print("causal mask:", "off" if args.no_causal_mask else "on")
    print("head 0 attention weights:")
    print(weights[0, 0].detach())


if __name__ == "__main__":
    main()
