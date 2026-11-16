import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Trainconfig
from .data import OFDMSymbolDataset, simulation_signature
from .model import NeuralReceiver
from .train import choose_device, run_epoch

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--frames",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    if args.frames < 1:
        raise ValueError("frames must be positive")

    device = choose_device()
    checkpoint_path = Path(args.checkpoint).resolve()

    # load将train中save的.pt文件读取到内存中以供使用
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint["simulation"] != simulation_signature():
        raise ValueError(
            "Simulation settings changed. Retrain the model."
        )
    # 保存了训练完成时的模型参数，将其赋给cfg
    cfg = Trainconfig(**checkpoint["config"])

    if cfg.mode != "normal":
        print(
            "DEBUG checkpoint: use normal training"
            "for final results."
        )

    model = NeuralReceiver(cfg.hidden_dim).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    print("Loaded epoch:",checkpoint["epoch"])

    ebn0_values_db = list(range(0, 21, 4))
    rows = []

    for index,ebn0_db in enumerate(ebn0_values_db):
        test_set = OFDMSymbolDataset(
            args.frames,
            (ebn0_db,ebn0_db),
            seed = cfg.seed + 1000 + index
        )

        loader = DataLoader(
            test_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        metrics = run_epoch(
            model,
            loader,
            device,
        )

        row = {
            "ebn0_db": ebn0_db,
            "ls_zf_ber": test_set.baseline_ber["LS + ZF"],
            "ls_mmse_ber": test_set.baseline_ber["LS + MMSE"],
            "nn_ber": metrics["ber"],
        }

        rows.append(row)

        print(
            f"Eb/N0={ebn0_db} dB | "
            f"LS+ZF={row['ls_zf_ber']:.6f} | "
            f"LS+MMSE={row['ls_mmse_ber']:.6f} | "
            f"NN={row['nn_ber']:.6f}",
            flush=True,
        )

    output_dir = checkpoint_path.parent

    with (output_dir / "test_ber.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f: 
        # 可以把f理解为从字典到csv的管道。
        # writer负责将字典转换到可以写入到csv中的值，f负责搬运。
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        # 写入表头
        writer.writeheader()
        # 自动根据表头填入值
        writer.writerows(rows)

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    # fig是整张画布，ax是画布上的一个子图
    fig, ax = plt.subplots(figsize = (7,5))

    for key, label, style in [
        ("ls_zf_ber", "LS + ZF", "b-o"),
        ("ls_mmse_ber", "LS + MMSE", "g--x"),
        ("nn_ber", "Neural receiver", "r-s"),
    ]:
        values = np.array(
            [row[key] for row in rows]
        )
        ax.semilogy(
            ebn0_values_db,
            np.where(values > 0, values, np.nan),
            style,
            label = label,
        )

    ax.set_xlabel("Data-bit Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.grid(True, which = "both")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir / "test_ber_1.png",
        dpi = 200,
    )

    plt.close(fig)

    print("Saved:", output_dir / "test_ber_1.png")

if __name__ == "__main__":
    main()

        