from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_prob = config.dropout_prob
        self.bias = config.bias
        self.initial_projection = nn.Linear(self.n_embd, 3*self.n_embd)
        self.output_projection = nn.Linear(self.n_embd, self.n_embd)
        self.out_dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        B, T, C = x.size()
        attention_keys = self.initial_projection(x) # attn keys is B,T,3C
        k, q, v = attention_keys.split(self.n_embd, dim=2) # each is B,T,C
        # scaled dot product attn expects B, nh, T, C//nh 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        # Y is B,nh,T,C//nh dimensional we want B,T,C
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.out_dropout(self.output_projection(y))
        return y




class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout_prob = config.dropout_prob
        self.bias = config.bias
        
        self.causal_self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.n_embd, bias=self.bias)
        self.layer_norm2 = nn.LayerNorm(self.n_embd, bias=self.bias)
        self.feed_forward = nn.Linear(self.n_embd, self.n_embd)


## config is a direct copy from karpathy's code
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout_prob: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
