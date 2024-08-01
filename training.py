from dataclasses import dataclass

import tiktoken
import torch
import torch.nn.functional as F

from model import GPT


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
#

class SingleBatchDataset(object):
    def __init__(self, config, device, batchsize=8):
        enc = tiktoken.get_encoding("gpt2")
        with open("./datasets/input.txt") as f:
            text = f.read()
        all_tokens = enc.encode(text)
        data = all_tokens[:config.block_size * batchsize + 1]
        self.x = torch.tensor(
            data[:-1], dtype=torch.long, device=device
        ).view(batchsize, -1)
        self.targets = torch.tensor(
            data[1:], dtype=torch.long, device=device
        ).view(batchsize, -1)

    def __iter__(self):
        return self

    def __next__(self):
        return self.x, self.targets


device = "cuda"
config = GPTConfig()

model = GPT(config)
data = SingleBatchDataset(config, device)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()
for x, targets in data:
    optimizer.zero_grad()
    logits, loss = model(x, targets)
    print(f"loss = {loss.item():.4f}")
    loss.backward()
    optimizer.step()
