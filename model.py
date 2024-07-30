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


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.n_embd
        self.dropout_prob = config.dropout_prob
        self.linear1 = nn.Linear(self.n_embed, 4 * self.n_embed)
        self.linear2 = nn.Linear(4 * self.n_embed, self.n_embed)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        return x


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.dropout_prob = config.dropout_prob
        self.bias = config.bias
        
        self.causal_self_attn = Attention(config)
        self.layer_norm1 = nn.LayerNorm(self.n_embd, bias=self.bias)
        self.layer_norm2 = nn.LayerNorm(self.n_embd, bias=self.bias)
        self.feed_forward = self.MLP(config)

    def forward(self, x):
        x = x + self.causal_self_attn(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.n_embed = config.n_embd
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.dropout_prob = config.dropout_prob
        self.bias = config.bias

        self.transformer = nn.ModuleDict(dict(
            input_embedding = nn.Embedding(self.vocab_size, self.n_embed),
            positional_embedding = nn.Embedding(self.block_size, self.n_embed),
            hidden_blocks = nn.ModuleList(
                [AttentionBlock(config) for _ in range(self.n_layer)]
            ),
            drop = nn.Dropout(self.dropout_prob),
            layer_norm = nn.LayerNorm(self.n_embed, self.bias)
        ))
        self.lm_head = nn.Linear(self.n_embed, self.vocab_size, bias=False)

        # tie the weights of language model head and embedding
        self.transformer.input_embedding.weight = self.lm_head.weight
        

    def forward(self, x):
        pass

