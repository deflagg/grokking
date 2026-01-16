import random

import torch
from torch.utils.data import DataLoader, Dataset


class ModularAdditionDataset(Dataset):
    def __init__(self, pairs, modulus):
        self.modulus = modulus
        self.eq_token = modulus
        inputs = []
        targets = []
        for x, y in pairs:
            inputs.append([x, y, self.eq_token])
            targets.append((x + y) % self.modulus)
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return self.targets.numel()

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_dataloaders(config):
    modulus = int(config["modulus"])
    seed = int(config.get("seed", 0))
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))

    pairs = [(x, y) for x in range(modulus) for y in range(modulus)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    split = len(pairs) // 2
    train_pairs = pairs[:split]
    test_pairs = pairs[split:]

    train_dataset = ModularAdditionDataset(train_pairs, modulus)
    test_dataset = ModularAdditionDataset(test_pairs, modulus)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    vocab_size = modulus + 1
    return train_loader, test_loader, vocab_size
