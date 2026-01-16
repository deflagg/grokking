import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, n_dims, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dims))
        self.bias = nn.Parameter(torch.zeros(n_dims)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, bias, block_size):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout, bias):
        super().__init__()
        self.fc = nn.Linear(d_model, d_ff, bias=bias)
        self.proj = nn.Linear(d_ff, d_model, bias=bias)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, bias, block_size):
        super().__init__()
        self.ln1 = LayerNorm(d_model, bias=bias)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            bias=bias,
            block_size=block_size,
        )
        self.ln2 = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=dropout, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layers,
        d_model,
        n_heads,
        d_ff,
        dropout,
        bias,
        tie_weights,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    bias=bias,
                    block_size=block_size,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = LayerNorm(d_model, bias=bias)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=bias)

        self.apply(self._init_weights)
        if tie_weights:
            self.lm_head.weight = self.token_embed.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        batch_size, seq_len = idx.size()
        if seq_len > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        x = self.token_embed(idx) + self.pos_embed(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
