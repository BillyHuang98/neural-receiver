import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import week2_ofdm as ofdm

from .config import Trainconfig
from .data import OFDMSymbolDataset, simulation_signature
from .model import NeuralReceiver

ROOT = Path(__file__).resolve().parents[1]

def choose_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

def run_epoch(model,loader,device,optimizer = None):
    """训练的代码; 传入optimizer时训练; 不传时只评估"""

    training = optimizer is not None
    model.train(training)

    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    bit_errors = 0
    sample_count = 0

    # with是python的“上下文管理器”
    # 意思是“在下面的这个缩进区域内，遵循某种特定的规则”。出了缩进规则自动失效。
    with torch.set_grad_enabled(training):
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none = True)

            logits = model(features)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            # 取出每一行中最大的那个（dim=1的作用）
            predicted = logits.detach().argmax(dim = 1)

            # 类别编号还原为两个比特，再统计BER
            bit_errors += (
                (predicted // 2 != labels // 2).sum().item()
                + (predicted % 2 != labels % 2).sum().item()
            )

            sample_count += labels.size(0)
            loss_sum += loss.item() * labels.size(0)

    return{
        "loss": loss_sum / sample_count,
        "ber": bit_errors / (2 * sample_count),
    }

def main():
    parser = argparse.ArgumentParser()
    # 创建参数使得在终端输入命令时能直接调用
    parser.add_argument(
        "--mode",
        choices=["normal","smoke","overfit"],
        default="normal",
    )

    parser.add_argument(
        "--epochs",
        type = int,
        default= None,
    )

    args = parser.parse_args()
    cfg = Trainconfig(mode = args.mode)

    if args.mode == "smoke":
        cfg.train_frames = 64
        cfg.val_frames = 16
        cfg.epochs = 2

    elif args.mode == "overfit":
        cfg.train_frames = 4
        cfg.val_frames = 4
        cfg.epochs = 300
        cfg.ebn0_max = 30.0
        cfg.ebn0_min = 30.0

    if args.epochs is not None:
        cfg.epochs = args.epochs

    if cfg.epochs < 1:
        raise ValueError("epochs must be positive")

    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    torch.set_num_threads(
        min(4,torch.get_num_threads())
    )

    ofdm.check_no_noise()

    device = choose_device()

    print("Device:",device,flush=True)
    print("Generating datasets...",flush=True)

    snr_range = (cfg.ebn0_min, cfg.ebn0_max)
    
    train_set = OFDMSymbolDataset(
        cfg.train_frames,
        snr_range,
        seed = cfg.seed + 1,
    )

    if cfg.mode == "overfit":
        val_set = train_set
    
        print("DEBUG: train and validation use the SAME data.")
    else:
        val_set = OFDMSymbolDataset(
            cfg.val_frames,
            snr_range,
            seed = cfg.seed + 2,
        )

    generator = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_set,
        batch_size= cfg.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size= cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Train samples:",len(train_set))
    print("Validation samples",len(val_set))
    print("validation baseline BER",val_set.baseline_ber)

    # to前可以填张量(line42-43)或者神经网络模块(line165)
    model = NeuralReceiver(cfg.hidden_dim).to(device)

    optimizer = torch.optim.AdamW(
        # 将之前类里面定义好的参数交给优化器
        model.parameters(),
        lr = cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = ROOT / "runs" / f"{cfg.mode}_{stamp}"
    run_dir.mkdir(parents=True,exist_ok=False)

    # 以下操作用于将初始配置、每一轮训练的数据写入json中
    with (run_dir / "config.json").open(
        "w",
        encoding= "utf-8",
    ) as f:
        json.dump(asdict(cfg),f,indent=2)

    best_loss = float("inf")

    with(run_dir / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        # 写入表头
        writer.writerow([
            "epoch",
            "train_loss",
            "train_ber",
            "val_loss",
            "val_ber",
        ])
        # 训练正式开始
        for epoch in range(1, cfg.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
            )

            val_metrics = run_epoch(
                model,
                val_loader,
                device,
            )

            writer.writerow([
                epoch,
                train_metrics["loss"],
                train_metrics["ber"],
                val_metrics["loss"],
                val_metrics["ber"],
            ])

            f.flush()

            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | "
                f"train loss={train_metrics['loss']:.4f} "
                f"BER={train_metrics['ber']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"BER={val_metrics['ber']:.4f}",
                flush=True,
            )

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": asdict(cfg),
                        "simulation": simulation_signature(),
                        "epoch": epoch,
                        "val_loss": val_metrics["loss"],
                        "val_ber": val_metrics["ber"],
                    },
                    run_dir / "best.pt",
                )
    print("Results:",run_dir)
    # 打印一个可以直接在终端中运行测试代码的指令
    print("Evaluate with:")

    print(
        f'& "{sys.executable}" -m week3.evaluate '
        # checkpoint是传入的参数，evaluate中会定义
        f'--checkpoint "{run_dir / "best.pt"}"'
    )

if __name__ == "__main__":
    main()