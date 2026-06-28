from __future__ import annotations

import argparse
import torch

from .train import make_loaders, evaluate
from .model import CRNN
from .data_loader import POLARITY_TO_ID


def run_eval(checkpoint: str, split: str, batch_size: int = 16, num_workers: int = 0, hidden_size: int = 192, dropout: float = 0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(batch_size=batch_size, num_workers=num_workers)
    loader = {"train": loaders[0], "val": loaders[1], "test": loaders[2]}[split]

    model = CRNN(n_mels=80, n_classes=len(POLARITY_TO_ID), hidden_size=hidden_size, dropout=dropout).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = evaluate(model, loader, criterion, device)
    print(f"{split}_loss={loss:.4f} {split}_acc={acc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="crnn_best.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()
    run_eval(args.checkpoint, args.split, args.batch_size, args.num_workers, args.hidden_size, args.dropout)


if __name__ == "__main__":
    main()


__all__ = ["run_eval"]
