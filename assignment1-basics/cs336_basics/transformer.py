import math
import torch
import torch.nn as nn
from einops import einsum, rearrange
from numpy.ma.core import size


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
        self.g = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms_x = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / self.d_model + self.eps)

        return (x / rms_x * self.g).to(in_type)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff:
            self.df = d_ff
        else:
            # Align up to a multiple of 64
            self.d_ff = int((d_model * 8 / 3 + 63) // 64 * 64)

        self.weight1 = Linear(d_model, d_ff, device=None, dtype=dtype)
        self.weight2 = Linear(d_ff, d_model, device=None, dtype=dtype)
        self.weight3 = Linear(d_model, d_ff, device=None, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.weight1(x)
        return self.weight2(w1x * torch.sigmoid(w1x) * self.weight3(x))


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
        precomputed_sines = torch.empty(max_seq_len * self.k_dim, device=device)
        precomputed_cosines = torch.empty(max_seq_len * self.k_dim, device=device)

        for i in range(self.max_seq_len):
            for k in range(self.k_dim):
                index = i * self.k_dim + k
                theta = get_theta_ik(self.theta, self.d_k, i, k+1)
                precomputed_sines[index] = math.sin(theta)
                precomputed_cosines[index] = math.cos(theta)

        self.register_buffer('precomputed_sines', precomputed_sines, persistent=False)
        self.register_buffer('precomputed_cosines', precomputed_cosines, persistent=False)

        rot_mat = self.get_rotation_matrix(0, 1)
        print(rot_mat.shape)
        print(rot_mat)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.d_k
        seq_len = x.size(-2)
        for i in range(seq_len):
            for k in range(self.k_dim):
                token_pos = token_positions[i]
                rot_mat = self.get_rotation_matrix(token_pos, k+1)
                x[::, 2 * k : 2 * (k+1)]  = x[2 * k : 2 * (k+1)] @ rot_mat
        return x

    def get_rotation_matrix(self, i, k) -> torch.Tensor:
        index = i * self.k_dim + k
        sine = self.precomputed_sines[index]
        cosine = self.precomputed_cosine[index]
        return torch.tensor(
            (cosine, -sine,
             sine, cosine)
        )










