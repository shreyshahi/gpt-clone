from dataclasses import dataclass

import tiktoken
import torch
import torch.nn.functional as F

from model import GPT


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout_prob: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


model = GPT(GPTConfig())
model.to("cuda")
model.eval()

prompt = input("enter prompt>")

enc = tiktoken.get_encoding("gpt2")
x = torch.tensor(enc.encode(prompt))

num_samples = 5
num_extra_tokens = 50

x = x.repeat(5, 1)
x = x.to("cuda")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

for _ in range(num_extra_tokens):
    logits, _ = model(x)
    top50, _ = torch.topk(logits, 50, dim=1)
    logits[logits < top50[:, [-1]]] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    sample = torch.multinomial(probs, num_samples=1)
    x = torch.cat((x, sample), dim=1)

for i in range(num_samples):
    print(f"sample {i}: {enc.decode(x[i].tolist())}")
