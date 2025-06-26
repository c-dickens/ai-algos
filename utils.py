import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Return accuracy of *model* on ``dataloader``.

    The dataloader may optionally yield sample weights alongside each batch.
    These weights are ignored for accuracy computation.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating accuracy", leave=False):
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits[:, -1, :].argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0
