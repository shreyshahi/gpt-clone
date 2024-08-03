import torch
import json
import tiktoken

import torch.nn.functional as F

def perform_validation(model, device, val_set):
    validation_loss = 0
    with torch.no_grad():
        for _ in range(20):
            x, targets = next(val_set)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, targets)
            validation_loss += loss.detach()
    return validation_loss / 20



def hellaswag_iterator(device):
    enc = tiktoken.get_encoding("gpt2")
    with open("./datasets/hellaswag/hellaswag_val.jsonl") as f:
        for line in f:
            data = json.loads(line)
            context = enc.encode(data["ctx"])
            label = data["label"]
            endings = [enc.encode(t) for t in data["endings"]]
            context_len = len(context)
            endings_lens = [len(ending) for ending in endings]
            tokens = torch.zeros(
                (4, context_len + max(endings_lens)),
                dtype=torch.long
            )
            masks = torch.zeros(
                (4, context_len + max(endings_lens)),
                dtype=torch.long
            )
            for i in range(4):
                tokens[i,:(context_len + endings_lens[i])] = torch.tensor(context + endings[i])
                masks[i, :(context_len + endings_lens[i])] = torch.tensor([0]*context_len + [1]*endings_lens[i])

            yield tokens.to(device), masks.to(device), label

def get_most_likely_row(tokens, mask, logits):
    """
    Straight copy paste from karpathy's code
    https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L258
    """
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm


def evaluate_hellaswag(model, device):
    num_total = 0
    num_correct = 0
    for tokens, masks, label in hellaswag_iterator(device):
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(tokens)
        pred = get_most_likely_row(tokens, masks, logits)
        num_total += 1
        num_correct += int(pred == label)
    return num_correct / num_total

