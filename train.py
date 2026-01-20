import getpass
import os
import random

import torch
import torch.nn.functional as F
import yaml
from dotenv import load_dotenv
import wandb

from data import make_dataloaders
from model import GPT


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)[:, -1, :]
            loss = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()
    model.train()
    return total_loss / total_count, total_correct / total_count


def main():
    load_dotenv()
    config = load_config("config.yaml")

    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_project:
        wandb_project = input("Enter WANDB project name: ").strip()
        if not wandb_project:
            raise RuntimeError("WANDB_PROJECT must be set in the environment.")
    if not wandb_api_key:
        wandb_api_key = getpass.getpass(
            "Enter WANDB API key (leave blank to skip login): "
        ).strip()
        if not wandb_api_key:
            wandb_api_key = None
    if wandb_api_key:
        wandb.login(key=wandb_api_key)

    run = wandb.init(project=wandb_project, config=config)

    seed = int(config.get("seed", 0))
    set_seed(seed)

    device = config.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader, vocab_size = make_dataloaders(config)

    d_model = int(config["d_model"])
    n_layers = int(config["n_layers"])
    n_heads = int(config["n_heads"])
    d_ff = int(config.get("d_ff", 4 * d_model))
    dropout = float(config.get("dropout", 0.0))
    bias = bool(config.get("bias", True))
    tie_weights = bool(config.get("tie_weights", False))
    block_size = int(config["block_size"])

    model = GPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        bias=bias,
        tie_weights=tie_weights,
    ).to(device)

    lr = float(config["lr"])
    weight_decay = float(config["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_steps = int(config["max_steps"])
    log_every = int(config.get("log_every", 1))
    train_iter = cycle(train_loader)

    for step in range(1, max_steps + 1):
        inputs, targets = next(train_iter)
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)[:, -1, :]
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            train_acc = (preds == targets).float().mean().item()

        if step % log_every == 0:
            test_loss, test_acc = evaluate(model, test_loader, device)
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/acc": train_acc,
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                },
                step=step,
            )

    save_path = config.get("save_path", "model.pt")
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    run.finish()


if __name__ == "__main__":
    main()
