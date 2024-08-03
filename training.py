from dataclasses import dataclass

import datasets
import math
import tiktoken
import time
import torch

from model import GPT
from validation import perform_validation, evaluate_hellaswag

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


class FineWebEduDataset(object):
    def __init__(self, config, device, split, batchsize=B):
        """
        first 50000 documents reserved for validation. Rest for training
        """
        self.split = split
        self.device = device
        assert (split == "train") or (split == "val")
        self.B = batchsize
        self.T = config.block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']
        data_save_path = "/mnt/datasets/fineweb-edu"

        self.dataset = datasets.load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            cache_dir=data_save_path
        )

        self.max_dataset_index = len(self.dataset["train"]) - 1
        self.current_dataset_index = 0
        if split == "train":
            self.current_dataset_index = 50000
        self.current_document_position = 0
        self.current_doc = self._load_next_doc()
        self.current_buffer = []

    def _load_next_doc(self):
        text = self.dataset["train"][self.current_dataset_index]["text"]
        tokens = [self.eot]
        tokens.extend(self.enc.encode(text))
        self.current_dataset_index += 1
        if self.split == "train":
            if self.current_dataset_index > self.max_dataset_index:
                self.current_dataset_index = 50000
        if self.split == "val":
            if self.current_dataset_index > 49999:
                self.current_dataset_index = 0
        return tokens

    def _get_next_buffer(self):
        B = self.B
        T = self.T
        cur_doc_pos = self.current_document_position
        cur_len = len(self.current_buffer)
        if cur_doc_pos + B * T + 1 - cur_len> len(self.current_doc):
            self.current_buffer.extend(self.current_doc[cur_doc_pos:])
            self.current_doc = self._load_next_doc()
            self.current_document_position = 0
            self._get_next_buffer()
        else:
            self.current_buffer.extend(
                self.current_doc[
                    cur_doc_pos:(cur_doc_pos + B * T + 1 - cur_len)
                ]
            )
            self.current_document_position += B * T + 1 - cur_len

    def __iter__(self):
        return self

    def __next__(self):
        self._get_next_buffer()
        buff = torch.tensor(
            self.current_buffer[:], dtype=torch.long
        ).to(self.device)
        x = buff[:-1].view(self.B, self.T)
        y = buff[1:].view(self.B, self.T)
        self.current_buffer = []
        return x, y

max_lr = 6e-4
min_lr = 0.1 * max_lr
warmup_steps = 10
max_steps=50
def get_lr(step):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    scale = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + scale * (max_lr - min_lr)


device = "cuda"

torch.set_float32_matmul_precision("high")

config = GPTConfig()

model = GPT(config)
data = FineWebEduDataset(config, device, "train")
val_set = FineWebEduDataset(config, device, "val")

model.to(device)
model = torch.compile(model)

decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

num_decay_params = sum(p.numel() for p in decay_params)
num_no_decay_params = sum(p.numel() for p in no_decay_params)

print(f"Num decayed tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
print(f"Num non decayed tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters")

optim_groups = [
    {"params": decay_params, "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0}
]
optimizer = torch.optim.AdamW(
    optim_groups,
    lr=6e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    fused=True,
)

full_batch_size = 524288
num_micro_batches = full_batch_size // (B * T)

model.train()
for i in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_iter in range(num_micro_batches):
        x, targets = next(data)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, targets)
        loss = loss / num_micro_batches
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = B * T * num_micro_batches / (t1 - t0)
    print(f"step: {i} | loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    if i % 5 == 0:
        validation_loss = perform_validation(model, device, val_set)
        print(f"VAL | step {i} | loss: {validation_loss:.4f}")
        hellaswag_score = evaluate_hellaswag(model, device)
        print(f"AWAG | step {i} | loss: {hellaswag_score:.4f}")
    if i > 50:
        break
