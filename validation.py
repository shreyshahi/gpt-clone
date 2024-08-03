import torch


def perform_validation(model, device, val_set):
    validation_loss = 0
    with torch.no_grad():
        for i in range(20):
            x, targets = next(val_set)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, targets)
            validation_loss += loss.detach()
    return validation_loss / 20



def evaluate_hellaswag(model, device):
    pass
