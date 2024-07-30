from dataclasses import dataclass

import torch



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



# plan for training
# 1) build a dataloader for one batch of tinyshakespear
# 2) loop on just one batch over and over and see if we can overfit it
