import torch
from torch import nn
from flash_attn import flash_attn_varlen_qkvpacked_func

from activation_checkpoint import checkpoint


class MultiheadSelfAttention(nn.Module):

    def __init__(self, num_channels, head_dim):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim

        self.to_query = nn.Linear(num_channels, num_channels, bias=True)
        self.to_key = nn.Linear(num_channels, num_channels, bias=True)
        self.to_value = nn.Linear(num_channels, num_channels, bias=True)
        self.query_norm = nn.RMSNorm(head_dim)
        self.key_norm = nn.RMSNorm(head_dim)

        self.out_layer = nn.Linear(num_channels, num_channels, bias=True)

    def forward(self, x, cu_seqlens):
        shape = x.shape[:-1]
        query, key, value = self.to_query(x), self.to_key(x), self.to_value(x)
        query = self.query_norm(query.reshape(*shape, self.num_heads, -1)).type_as(
            query
        )
        key = self.key_norm(key.reshape(*shape, self.num_heads, -1)).type_as(key)
        value = value.reshape(*shape, self.num_heads, -1)
        query_key_value = torch.stack([query, key, value], dim=-3)

        out = flash_attn_varlen_qkvpacked_func(
            query_key_value, cu_seqlens, torch.diff(cu_seqlens).max()
        ).flatten(-2, -1)
        out = self.out_layer(out)
        return out


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim):
        super().__init__()
        self.in_layer = nn.Linear(dim, ff_dim, bias=False)
        self.activation = nn.GELU()
        self.out_layer = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(self.activation(self.in_layer(x)))


class ModelBlock(nn.Module):

    def __init__(self, dim, ff_dim, head_dim):
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = MultiheadSelfAttention(dim, head_dim)

        self.feed_forward_norm = nn.RMSNorm(dim)
        self.feed_forward = FeedForward(dim, ff_dim)

    def forward(self, x, cu_seqlens):
        x = x + self.attention(self.attention_norm(x), cu_seqlens)
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, ff_dim, num_layers, head_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ModelBlock(hidden_dim, ff_dim, head_dim) for _ in range(num_layers)]
        )
        self.out_layer = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, cu_seqlens):
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x, cu_seqlens)
        return self.out_layer(x)


class ModelBlockCheckpoint(nn.Module):
    def __init__(self, dim, ff_dim, head_dim, use_rng_state=True):
        super().__init__()
        self.use_rng_state = use_rng_state
        self.attention_norm = nn.RMSNorm(dim)
        self.attention = MultiheadSelfAttention(dim, head_dim)

        self.feed_forward_norm = nn.RMSNorm(dim)
        self.feed_forward = FeedForward(dim, ff_dim)

    def forward(self, x, cu_seqlens):
        def block_fn(x_, cu_, *args):
            x1 = x_ + self.attention(self.attention_norm(x_), cu_)
            return x1 + self.feed_forward(self.feed_forward_norm(x1))

        return checkpoint(block_fn, x, cu_seqlens, self.use_rng_state)


class ModelCheckpoint(nn.Module):
    def __init__(self, in_dim, hidden_dim, ff_dim, num_layers, head_dim, use_rng_state=True):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                ModelBlockCheckpoint(hidden_dim, ff_dim, head_dim, use_rng_state)
                for _ in range(num_layers)
            ]
        )
        self.out_layer = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, cu_seqlens):
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x, cu_seqlens)
        return self.out_layer(x)


class LMModel(nn.Module):
    def __init__(self, vocab_size, dim, ff_dim, num_layers, head_dim, checkpointing: bool = True, use_rng_state: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        blocks = []
        for _ in range(num_layers):
            if checkpointing:
                blocks.append(ModelBlockCheckpoint(dim, ff_dim, head_dim, use_rng_state))
            else:
                blocks.append(ModelBlock(dim, ff_dim, head_dim))

        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight # weight tie

    def forward(self, input_ids, cu_seqlens):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, cu_seqlens)
            x = self.norm(x)
        return self.lm_head(x)