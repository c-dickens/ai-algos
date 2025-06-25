import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader

def evaluate_accuracy(model:nn.Module, dataloader:DataLoader, device:torch.device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating accuracy"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            last_token_logits = logits[:, -1, :]  # (B, 2)
            preds = last_token_logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0
