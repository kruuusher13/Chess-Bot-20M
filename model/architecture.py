import torch
import torch.nn as nn

class ChessEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len) of token IDs
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(x) + self.pos_emb(positions)
        return self.dropout(self.norm(x))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V and reshape for multi-head
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        # attn = softmax(Q K^T / sqrt(d_k)) V
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.W_o(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # Pre-norm: normalize BEFORE the sublayer (more stable training)
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class ChessTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 seq_len, n_moves, dropout=0.1):
        super().__init__()
        self.embedding = ChessEmbedding(vocab_size, d_model, seq_len, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_moves)

    def forward(self, x):
        # x: (batch, 80) token IDs
        x = self.embedding(x)          # (batch, 80, 512)
        for block in self.blocks:
            x = block(x)               # (batch, 80, 512)
        x = self.final_norm(x)
        cls = x[:, 0, :]               # (batch, 512) — CLS token
        logits = self.classifier(cls)   # (batch, 1968)
        return logits
