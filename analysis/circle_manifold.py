import argparse
import os
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import GPT


def parse_value(raw_value):
    value = raw_value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path):
    config = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            if not key:
                continue
            config[key] = parse_value(raw_value)
    return config


def resolve_device(config):
    device = str(config.get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_model(config):
    modulus = int(config["modulus"])
    vocab_size = int(config.get("vocab_size", modulus + 1))
    block_size = int(config["block_size"])
    n_layers = int(config["n_layers"])
    d_model = int(config["d_model"])
    n_heads = int(config["n_heads"])
    d_ff = int(config.get("d_ff", 4 * d_model))
    dropout = float(config.get("dropout", 0.0))
    bias = bool(config.get("bias", True))
    tie_weights = bool(config.get("tie_weights", False))

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
    )
    return model, modulus


def load_state_dict(model, ckpt_path, device):
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"error: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"error: failed to load checkpoint {ckpt_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        print(
            f"error: state_dict mismatch when loading {ckpt_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def build_probe_inputs(modulus, probe, device):
    eq_token = modulus
    a = torch.arange(modulus, dtype=torch.long, device=device)
    zeros = torch.zeros(modulus, dtype=torch.long, device=device)
    eqs = torch.full((modulus,), eq_token, dtype=torch.long, device=device)
    if probe == "a0eq":
        batch = torch.stack([a, zeros, eqs], dim=1)
    elif probe == "0aeq":
        batch = torch.stack([zeros, a, eqs], dim=1)
    else:
        raise ValueError(f"Unknown probe pattern: {probe}")
    return batch


def capture_residuals(model, inputs, layer, pos):
    activations = {}
    handles = []

    def save_hook(name):
        def hook(_module, _inputs, output):
            activations[name] = output.detach().cpu()

        return hook

    handles.append(model.drop.register_forward_hook(save_hook("embed")))
    for idx, block in enumerate(model.blocks):
        handles.append(block.register_forward_hook(save_hook(f"block_{idx}")))
    handles.append(model.ln_f.register_forward_hook(save_hook("ln_f")))

    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    for handle in handles:
        handle.remove()

    if layer == -1:
        key = "ln_f"
    elif layer == -2:
        key = "embed"
    else:
        key = f"block_{layer}"

    if key not in activations:
        raise KeyError(f"Activation '{key}' not captured. Check layer index.")

    act = activations[key]
    if pos < 0 or pos >= act.shape[1]:
        raise ValueError(f"pos {pos} out of range for sequence length {act.shape[1]}")

    return act[:, pos, :].numpy()


def project_pca(vectors):
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    return coords[:, 0], coords[:, 1]


def select_fourier_k(vectors):
    p = vectors.shape[0]
    indices = np.arange(p)
    best_k = None
    best_norm = -1.0
    max_k = (p - 1) // 2
    for k in range(1, max_k + 1):
        phase = np.exp(-2j * np.pi * k * indices / p)
        rk = (phase[:, None] * vectors).mean(axis=0)
        rk_norm = np.linalg.norm(rk)
        if rk_norm > best_norm:
            best_norm = rk_norm
            best_k = k
    return best_k


def project_fourier(vectors, k):
    p = vectors.shape[0]
    indices = np.arange(p)
    phase = np.exp(-2j * np.pi * k * indices / p)
    rk = (phase[:, None] * vectors).mean(axis=0)
    u = np.real(rk)
    v = np.imag(rk)
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    if u_norm < 1e-12 or v_norm < 1e-12:
        raise ValueError("Fourier plane vectors are degenerate.")
    u = u / u_norm
    v = v / v_norm
    x = vectors @ u
    y = vectors @ v
    return x, y


def plot_circle(x, y, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(x, y, s=18)
    loop_x = np.append(x, x[0])
    loop_y = np.append(y, y[0])
    ax.plot(loop_x, loop_y, linewidth=0.8, alpha=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_k(value):
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid k: {value}") from exc


def main():
    parser = argparse.ArgumentParser(description="Circle manifold probe visualizer.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--ckpt", default=None, help="Path to model checkpoint")
    parser.add_argument("--space", required=True, choices=["embed", "resid"])
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pos", type=int, default=2)
    parser.add_argument("--probe", default="a0eq", choices=["a0eq", "0aeq"])
    parser.add_argument("--method", required=True, choices=["pca", "fourier"])
    parser.add_argument("--k", type=parse_k, default="auto")
    parser.add_argument("--outdir", default=os.path.join("analysis", "outputs"))

    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)

    model, modulus = build_model(config)
    model.to(device)

    ckpt_path = args.ckpt or config.get("save_path", "model.pt")
    load_state_dict(model, ckpt_path, device)

    if args.space == "embed":
        vectors = model.token_embed.weight[:modulus].detach().cpu().numpy()
    else:
        inputs = build_probe_inputs(modulus, args.probe, device)
        vectors = capture_residuals(model, inputs, args.layer, args.pos)

    vectors = vectors.astype(np.float64, copy=False)

    if args.method == "pca":
        x, y = project_pca(vectors)
        k_used = args.k
    else:
        if args.k == "auto":
            k_used = select_fourier_k(vectors)
        else:
            k_used = args.k
        x, y = project_fourier(vectors, k_used)

    os.makedirs(args.outdir, exist_ok=True)
    k_label = "auto" if args.k == "auto" else str(args.k)
    out_name = (
        f"{args.space}_{args.method}_layer{args.layer}_pos{args.pos}"
        f"_probe{args.probe}_k{k_label}.png"
    )
    out_path = os.path.join(args.outdir, out_name)

    if args.method == "fourier" and args.k == "auto":
        title_k = f"auto({k_used})"
    else:
        title_k = str(k_used)
    title = (
        f"space={args.space} method={args.method} layer={args.layer} "
        f"pos={args.pos} probe={args.probe} k={title_k}"
    )
    plot_circle(x, y, title, out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
