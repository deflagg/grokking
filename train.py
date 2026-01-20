import csv
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


def init_csv_logger(log_path):
    if not log_path:
        return None, None
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    writer.writerow(["step", "train_loss", "train_acc", "val_loss", "val_acc"])
    log_file.flush()
    return log_file, writer


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
    log_path = config.get("log_path", "")
    log_file, log_writer = init_csv_logger(log_path)
    train_iter = cycle(train_loader)

    try:
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
                val_loss, val_acc = evaluate(model, test_loader, device)
                train_loss = loss.item()
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "test/loss": val_loss,
                        "test/acc": val_acc,
                    },
                    step=step,
                )
                print(
                    f"step {step} "
                    f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
                    f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}",
                    flush=True,
                )
                if log_writer:
                    log_writer.writerow(
                        [step, train_loss, train_acc, val_loss, val_acc]
                    )
                    log_file.flush()
    finally:
        if log_file:
            log_file.close()

    save_path = config.get("save_path", "model.pt")
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), save_path)

    run.finish()


if __name__ == "__main__":
    main()
