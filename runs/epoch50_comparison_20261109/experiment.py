"""Compare 20 and 50 epochs without changing the project's source files.

Run from the project root with: D:\\python\\python.exe -B runs/epoch50_comparison_20261109/experiment.py
"""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import week2_ofdm as ofdm
from week1_qpsk_awgn import qpsk_demodulate
from week3.config import Trainconfig
from week3.data import OFDMSymbolDataset, simulation_signature
from week3.model import NeuralReceiver
from week3.train import choose_device, run_epoch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})


def write_json(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if (OUTPUT / "history.csv").exists():
        raise RuntimeError("This experiment already has results; use a fresh output directory.")
    protected = {
        str(p.relative_to(ROOT)): digest(p)
        for p in ROOT.rglob("*")
        if p.is_file()
        and OUTPUT not in p.parents
        and (p.suffix == ".py" or (ROOT / "runs" in p.parents))
    }
    write_json("original_files_sha256.json", protected)
    baseline_dir = ROOT / "runs" / "normal_20261109_143827_233212"
    old_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    cfg = Trainconfig(**old_config)
    cfg.epochs = 50
    write_json("config.json", asdict(cfg))
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    ofdm.check_no_noise()
    device = choose_device()
    print("Output:", OUTPUT, flush=True)
    print("Device:", device, "| Training range:", (cfg.ebn0_min, cfg.ebn0_max), flush=True)
    train_set = OFDMSymbolDataset(cfg.train_frames, (cfg.ebn0_min, cfg.ebn0_max), cfg.seed + 1)
    val_set = OFDMSymbolDataset(cfg.val_frames, (cfg.ebn0_min, cfg.ebn0_max), cfg.seed + 2)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(cfg.seed), num_workers=0)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    model = NeuralReceiver(cfg.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    best_loss = float("inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        history.append(dict(epoch=epoch, train_loss=train["loss"], train_ber=train["ber"],
                            val_loss=val["loss"], val_ber=val["ber"]))
        write_csv("history.csv", history)
        checkpoint = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                          config=asdict(cfg), simulation=simulation_signature(), epoch=epoch,
                          val_loss=val["loss"], val_ber=val["ber"])
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            torch.save(checkpoint, OUTPUT / "best.pt")
        if epoch in (20, 50):
            torch.save(checkpoint, OUTPUT / f"epoch{epoch}.pt")
        print(f"Epoch {epoch:02d}/50 | train loss={train['loss']:.6f} BER={train['ber']:.6f} | "
              f"val loss={val['loss']:.6f} BER={val['ber']:.6f}", flush=True)

    with (baseline_dir / "history.csv").open(encoding="utf-8", newline="") as f:
        old_history = list(csv.DictReader(f))
    prefix_difference = max(abs(float(old[key]) - float(new[key]))
                            for old, new in zip(old_history, history[:20])
                            for key in ("train_loss", "train_ber", "val_loss", "val_ber"))
    models = {}
    epochs = {}
    checkpoints = {
        "best20": baseline_dir / "best.pt",
        "epoch20": OUTPUT / "epoch20.pt",
        "best50": OUTPUT / "best.pt",
        "epoch50": OUTPUT / "epoch50.pt",
    }
    for name, path in checkpoints.items():
        ck = torch.load(path, map_location="cpu", weights_only=True)
        assert ck["simulation"] == simulation_signature()
        net = NeuralReceiver(cfg.hidden_dim)
        net.load_state_dict(ck["model_state"])
        models[name] = net.eval()
        epochs[name] = ck["epoch"]
    print("Checkpoint epochs:", epochs, "| First-20 history max difference:", prefix_difference, flush=True)

    comparisons = []
    standard = []
    for index, snr in enumerate(range(0, 41, 4)):
        data = OFDMSymbolDataset(1000, (snr, snr), cfg.seed + 1000 + index)
        loader = DataLoader(data, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        row = dict(ebn0_db=snr, total_bits=2 * len(data), ls_zf_ber=data.baseline_ber["LS + ZF"],
                   ls_mmse_ber=data.baseline_ber["LS + MMSE"])
        for name, net in models.items():
            row[name + "_ber"] = run_epoch(net, loader, torch.device("cpu"))["ber"]
        comparisons.append(row)
        standard.append(dict(ebn0_db=snr, ls_zf_ber=row["ls_zf_ber"],
                             ls_mmse_ber=row["ls_mmse_ber"], nn_ber=row["best50_ber"]))
        print("Sweep:", json.dumps(row), flush=True)
    write_csv("comparison_ber.csv", comparisons)
    write_csv("test_ber.csv", standard)

    diagnostics = []
    fading_details = []
    for condition, noise_var in (("40dB", 1 / (2 * 10**4)), ("no_noise", 0.0)):
        rng = np.random.default_rng(20261109)
        features, labels, zf_bits, magnitudes = [], [], [], []
        for _ in range(10000):
            bits, _, _, tx = ofdm.make_frame(rng)
            rx, h_true = ofdm.multipath_channel(tx, noise_var, rng)
            y_freq = ofdm.ofdm_demodulate(rx)
            h_ls = ofdm.estimate_channel_ls(y_freq)
            y, h = y_freq[ofdm.DATA_INDEX], h_ls[ofdm.DATA_INDEX]
            features.append(np.column_stack([y.real, y.imag, h.real, h.imag]).astype(np.float32))
            labels.append(bits.reshape(-1, 2))
            zf_bits.append(qpsk_demodulate(ofdm.equalize_zf(y_freq, h_ls)[ofdm.DATA_INDEX]).reshape(-1, 2))
            magnitudes.append(np.abs(h_true[ofdm.DATA_INDEX]))
        features = torch.from_numpy(np.concatenate(features))
        labels = np.concatenate(labels)
        magnitudes = np.concatenate(magnitudes)
        errors = {"ls_zf": np.count_nonzero(np.concatenate(zf_bits) != labels, axis=1)}
        with torch.inference_mode():
            for name, net in models.items():
                pred = torch.cat([net(batch).argmax(1) for batch in features.split(8192)]).numpy()
                predicted_bits = np.column_stack([pred // 2, pred % 2])
                errors[name] = np.count_nonzero(predicted_bits != labels, axis=1)
        row = dict(condition=condition, total_bits=int(labels.size))
        for name, values in errors.items():
            row[name + "_bit_errors"] = int(values.sum())
            row[name + "_ber"] = float(values.sum() / labels.size)
            for low, high in ((0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 1), (1, np.inf)):
                mask = (magnitudes >= low) & (magnitudes < high)
                fading_details.append(dict(condition=condition, model=name, h_min=low, h_max=high,
                                           symbols=int(mask.sum()), bit_errors=int(values[mask].sum())))
        diagnostics.append(row)
        print("Diagnostic:", json.dumps(row), flush=True)
    write_csv("diagnostics_960000bits.csv", diagnostics)
    write_csv("diagnostics_by_channel_magnitude.csv", fading_details)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in (("train_loss", "Training"), ("val_loss", "Validation")):
        axes[0].plot([h["epoch"] for h in history], [h[key] for h in history], label=label)
    axes[0].axvline(20, color="gray", linestyle=":")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    for key, label, style in (("ls_zf_ber", "LS + ZF / MMSE", "k-o"),
                              ("best20_ber", f"20-epoch budget: best epoch {epochs['best20']}", "b--s"),
                              ("best50_ber", f"50-epoch budget: best epoch {epochs['best50']}", "g--^"),
                              ("epoch50_ber", "Actual epoch 50", "r-x")):
        values = np.array([r[key] for r in comparisons])
        axes[1].semilogy([r["ebn0_db"] for r in comparisons],
                         np.where(values > 0, values, np.nan), style, label=label)
    axes[1].set(xlabel="Data-bit Eb/N0 (dB)", ylabel="BER")
    for ax in axes:
        ax.grid(True, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison.png", dpi=180)
    plt.close(fig)
    changed = [p for p, expected in protected.items()
               if not (ROOT / p).is_file() or digest(ROOT / p) != expected]
    summary = dict(training_config=asdict(cfg), checkpoint_epochs=epochs,
                   first20_history_max_abs_difference=prefix_difference,
                   diagnostics=diagnostics, original_files_changed=changed,
                   interpretation="Single training seed; best checkpoints selected by mixed-SNR validation loss. "
                   "Actual epoch 20 and epoch 50 are also evaluated to isolate extra training.")
    write_json("summary.json", summary)
    lines = ["# 20 versus 50 epochs", "", "Training range: 0-40 dB. All other training settings match the saved 20-epoch run.",
             f"First-20 history max absolute difference: {prefix_difference}.",
             f"Selected checkpoint epochs: {epochs}.", "", "Each diagnostic condition contains 960,000 bits.", "",
             "| Condition | ZF errors | Previous best (20-epoch budget) | Actual epoch 20 | Best (50-epoch budget) | Actual epoch 50 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in diagnostics:
        lines.append(f"| {r['condition']} | {r['ls_zf_bit_errors']} | {r['best20_bit_errors']} | "
                     f"{r['epoch20_bit_errors']} | {r['best50_bit_errors']} | {r['epoch50_bit_errors']} |")
    lines += ["", "The diagnostics use the same samples for all methods. This is one training seed.",
              "Source files, original checkpoints, and original results were verified by SHA-256.",
              f"Original files changed: {changed}.", "", "![Loss and BER comparison](comparison.png)", ""]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    assert not changed, f"Original files changed: {changed}"
    print("Complete. Original files unchanged. Results:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
