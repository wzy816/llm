import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.w * x


def apply_rotary_position_embedding(x, freqs):
    # last dimension [x0,x1,x2,..] to [[x0,x1],[x2,]..] pairs
    # (batch_size, seq_len, num_heads, head_dim/2, 2)
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    # (batch_size, seq_len, num_heads, head_dim/2)
    x = torch.view_as_complex(x)

    _, seq_len, _, half_head_dim = x.shape
    # (1, seq_len, 1,head_dim/2)
    freqs = freqs[0:seq_len].view(1, seq_len, 1, half_head_dim)

    o = torch.view_as_real(x * freqs)
    return o.flatten(3)


def compute_freqs(head_dim, max_seq_len):
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2)
                   [: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len*2)
    freqs = torch.outer(t, freqs).float()
    # (2*max_seq_len, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(4*dim * 2/3)
        hidden_dim = hidden_dim+256 - hidden_dim % 256  # multiple of 256
        self.w = nn.Linear(dim, hidden_dim, bias=False)
        self.v = nn.Linear(dim, hidden_dim, bias=False)
        self.w_2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w_2(F.silu(self.w(x))*self.v(x))


class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, freqs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.freqs = freqs

        self.wq = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.wo = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = apply_rotary_position_embedding(q, self.freqs)
        k = apply_rotary_position_embedding(k, self.freqs)

        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        # (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

        # mask by fill upper right with -inf
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf")).cuda()
        mask = torch.triu(mask, diagonal=1)
        scores = scores + mask  # (batch_size, num_heads, seq_len, seq_len)

        scores = F.softmax(scores.float(), dim=-1)

        o = torch.matmul(scores, v)
        o = o.transpose(1, 2).contiguous().view(
            batch_size, seq_len, dim)  # (batch_size, seq_len, dim)
        o = self.wo(o)
        return o


class Block(nn.Module):
    def __init__(self, dim, num_heads, head_dim, max_seq_len):
        super().__init__()
        self.norm_1 = RMSNorm(dim)
        self.attention = Attention(dim, num_heads, head_dim, max_seq_len)
        self.norm_2 = RMSNorm(dim)
        self.mlp = MLP(dim)

#     def forward(self, x):
#         x = self.norm_1(x + self.attention(x))
#         x = self.norm_2(x + self.mlp(x))
#         return x

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs
        return custom_forward
       
    def forward(self, x):
        x = x + checkpoint(self.custom(self.attention), x)
        x = self.norm_1(x)
        x = x + checkpoint(self.custom(self.mlp), x)
        x = self.norm_2(x)
        return x



class LLM(nn.Module):
    def __init__(self, vocab_size, padding_idx, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(vocab_size, config.dim, padding_idx)

        self.layers = nn.ModuleList()
        freqs = compute_freqs(config.head_dim, config.max_seq_len)
        for i in range(config.num_layers):
            block = Block(config.dim, config.num_heads,
                          config.head_dim, freqs)
            self.layers.append(block)

        self.norm = RMSNorm(config.dim)
        self.output = nn.Linear(config.dim, vocab_size, bias=False)

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # simplify 1/√N weight scaling in gpt2 paper
                factor = 1
                if name.endswith('wo'):
                    factor = 1 / math.sqrt(2*self.config.num_layers)

                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02*factor)
            if isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        o = self.output(h)
        return o.float()

    def count_parameters(self):
        total = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
                total += sum([p.numel() for p in list(module.parameters())])
        if total > 1e9:
            return f'{total/ 1e9:.2f}B'
        else:
            return f'{total/ 1e6:.2f}M'
