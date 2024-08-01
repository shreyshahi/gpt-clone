from dataclasses import dataclass

import tiktoken
import time
import torch
import torch.nn.functional as F

from model import GPT

torch.manual_seed(42)
torch.cuda.manual_seed(42)

B = 16
T = 1024

## config is a direct copy from karpathy's code
@dataclass
class GPTConfig:
    block_size: int = T
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
    def __init__(self, config, device, batchsize=B):
        enc = tiktoken.get_encoding("gpt2")
        with open("./datasets/input.txt") as f:
            text = f.read()
        all_tokens = enc.encode(text)
        data = torch.tensor(
            all_tokens[:config.block_size * batchsize + 1],
            dtype=torch.long,
            device=device,
        )
        self.x = data[:-1].view(batchsize, -1)
        self.targets = data[1:].view(batchsize, -1)

    def __iter__(self):
        return self

    def __next__(self):
        return self.x, self.targets

class TinyShakespeareDataset(object):
    def __init__(self, config, device, batchsize=B):
        enc = tiktoken.get_encoding("gpt2")
        with open("./datasets/input.txt") as f:
            text = f.read()
        all_tokens = enc.encode(text)
        self.data = torch.tensor(all_tokens, dtype=torch.long)
        self.current_pos = 0
        self.B = batchsize
        self.T = config.block_size
        self.device = device
        print(f"loaded {self.data.size(0)} tokens")
        print(f"We need {self.data.size(0) // (self.B * self.T)} epochs to train on all data")

    def __iter__(self):
        return self

    def __next__(self):
        buffer = self.data[
            self.current_pos:(self.current_pos + self.B * self.T + 1)
        ]
        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)
        self.current_pos += self.B * self.T
        if self.current_pos + self.B*self.T > self.data.size(0):
            self.current_pos = 0
        return(x.to(self.device), y.to(self.device))

device = "cuda"

torch.set_float32_matmul_precision("high")

config = GPTConfig()

model = GPT(config)
data = TinyShakespeareDataset(config, device)

model.to(device)
model = torch.compile(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()
for i, (x, targets) in enumerate(data):
    t0 = time.time()
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, targets)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = B * T / (t1 - t0)
    print(f"step: {i}, loss: {loss.item():.4f}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")
    if i > 50:
        break
