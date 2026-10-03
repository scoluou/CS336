import math
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim, device=device, dtype=dtype))
        sigma = math.sqrt(2 / (in_dim + out_dim))
        sigma3 = 3 * sigma
        torch.nn.init.trunc_normal_(self.weight, 0, sigma, -sigma3, sigma3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return rearrange(torch.index_select(self.weight, 0,
                                            rearrange(token_ids, 'batch seq_len -> (batch seq_len)')),
                         '(batch seq_len) d_embed -> batch seq_len d_embed',
                         batch=token_ids.size(0))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable 'gain' parameter
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms_x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / self.d_model + self.eps)

        return (x / rms_x * self.weight).to(in_type)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff:
            self.df = d_ff
        else:
            # Align up to a multiple of 64
            self.d_ff = int((d_model * 8 / 3 + 63) // 64 * 64)

        self.w1 = Linear(d_model, d_ff, device=None, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=None, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=None, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        return self.w2(w1x * torch.sigmoid(w1x) * self.w3(x))


# Rotary Position Embedding
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        def get_theta_ik(theta: float, d: int, i: int, k: int) -> float:
            return i / (pow(theta, (2 * k - 2) / d))

        self.k_dim = int(self.d_k / 2)
        precomputed_sines = torch.empty((max_seq_len, self.k_dim), device=device)
        precomputed_cosines = torch.empty((max_seq_len, self.k_dim), device=device)

        for i in range(self.max_seq_len):
            for k in range(self.k_dim):
                theta = get_theta_ik(self.theta, self.d_k, i, k+1)
                precomputed_sines[i, k] = math.sin(theta)
                precomputed_cosines[i, k] = math.cos(theta)

        self.register_buffer('precomputed_sines', precomputed_sines, persistent=False)
        self.register_buffer('precomputed_cosines', precomputed_cosines, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.d_k

        sin = self.precomputed_sines[token_positions]
        cos = self.precomputed_cosines[token_positions]
        x = rearrange(x, '... (k_dim pair) -> ... k_dim pair', pair=2)
        a, b = x.unbind(dim=-1)

        # [a, b] * R^T, note transpose of R here
        return rearrange(torch.stack(((a * cos - b * sin), a * sin + b * cos), dim=-1), '... k_dim pair -> ... (k_dim pair)')


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    exp = torch.exp(x - torch.max(x, dim=i, keepdim=True).values)
    return exp / torch.sum(exp, dim=i, keepdim=True)


def scaled_dot_production_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d_k = Q.size(-1)
    scaled_dot_production = einsum(Q, K, '... n d_k, ... m d_k -> ... n m') / math.sqrt(d_k)
    return einsum(softmax(scaled_dot_production.masked_fill(~mask, float('-inf')), -1), V, '... n m, ... m d_v -> ... n d_v')


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.q_proj = Linear(self.d_model, self.d_model, device)
        self.k_proj = Linear(self.d_model, self.d_model, device)
        self.v_proj = Linear(self.d_model, self.d_model, device)
        self.output_proj = Linear(self.d_model, self.d_model, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        K = rearrange(K, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        V = rearrange(V, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)

        multi_head_attn = scaled_dot_production_attention(Q, K, V, causal_mask)
        multi_head_attn = rearrange(multi_head_attn, '... h seq_len d_v -> ... seq_len (h d_v)')
        multi_head_attn = self.output_proj(multi_head_attn)

        return multi_head_attn


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.rope = RoPE(theta, self.d_k, max_seq_len, device)

        self.q_proj = Linear(self.d_model, self.d_model, device)
        self.k_proj = Linear(self.d_model, self.d_model, device)
        self.v_proj = Linear(self.d_model, self.d_model, device)
        self.output_proj = Linear(self.d_model, self.d_model, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.size(-2)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = rearrange(Q, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        K = rearrange(K, '... seq_len (h d_k) -> ... h seq_len d_k', h=self.num_heads)
        V = rearrange(V, '... seq_len (h d_v) -> ... h seq_len d_v', h=self.num_heads)

        # Apply rope on Q, K
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        multi_head_attn = scaled_dot_production_attention(Q, K, V, causal_mask)
        multi_head_attn = rearrange(multi_head_attn, '... h seq_len d_v -> ... seq_len (h d_v)')
        multi_head_attn = self.output_proj(multi_head_attn)

        return multi_head_attn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff:int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model, 1e-5, device)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta, max_seq_len, device)
        self.ffn = FeedForwardNetwork(d_model, d_ff, device)
        self.ln2 = RMSNorm(d_model, 1e-5, device)


    def forward(self, x: torch.Tensor, token_position: torch.Tensor | None = None) -> torch.Tensor:
        y = x + self.attn(self.ln1(x), token_position)

        return y + self.ffn(self.ln2(y))


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model:int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
                 device=None):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device) for i in range(num_layers)])
        self.ln_final = RMSNorm(d_model, 1e-5, device)
        self.lm_head = Linear(d_model, vocab_size, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        embedded = self.token_embeddings(x)

        for layer in self.layers:
            embedded = layer(embedded, token_positions)

        return self.lm_head(self.ln_final(embedded))
