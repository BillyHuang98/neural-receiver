import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import matplotlib
# 使得在无图形界面的终端上也可正常运行
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from week3.data import simulation_signature
from week3.model import NeuralReceiver
from .simulator import Scenario, make_data

ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "perfect_zf": ("perfect CSI + ZF", "k"),
    "ls_zf": ("LS + ZF", "b-o"),
    "ls_mmse": ("LS + MMSE", "g--x"),
    "nn": ("Neural receiver", "r-s"),
}

def make_scenarios(signature):
    """输入:之前定义的signature结构体
       输出:包含各类场景配置的scenarios数组
    """
    if signature["pilot_symbol"] != [1.0, 0.0]:
        raise ValueError("this simulator uses pilot symbol 1+0j")

    # 兼容之前代码中pilot_step / Pilot_steps 两种拼写
    step = signature.get(
        "pilot_step",
        signature.get("pilot_steps"),
    )

    base = Scenario(
        name="baseline",
        true_taps=signature["n_taps"],
        rx_taps=signature["n_taps"],
        pilot_step=step,
        n_fft=signature["n_fft"],
        n_cp=signature["n_cp"],
    )

    scenarios = [
        base,

        replace(
            base,
            name="taps2",
            true_taps = 2,
        ),

        replace(
            base,
            name="taps8",
            true_taps = 8,
        ),

        replace(
            base,
            name = "flat8",
            true_taps =8,
            profile = "uniform",
        ),

        replace(
            base,
            name="flat8_rx8",
            true_taps=8,
            rx_taps=8,
            profile="uniform",
        ),

        replace(
            base,
            name="pilot_dense",
            pilot_step=2,
        ),
 
        replace(
            base,
            name="pilot_sparse",
            pilot_step=8,
        ),
    ]

    for scenario in scenarios:
        scenario.validate()

    return scenarios

def evaluate_trial(model, device, scenario, snr, frames, seed):

    """在指定场景、信噪比和随机种子下完成单次仿真测试与算法性能对比。

    功能:
        1. 仿真生成当前条件下的 OFDM 频域数据与传统基线(ZF/MMSE)结果;
        2. 分批调用神经接收机完成前向推理与 QPSK 比特硬解调；
        3. 统计各算法在整批数据上的 BER,并计算 NN 与 LS+ZF 基于独立 OFDM 块的配对差值统计量。

    参数:
        model (torch.nn.Module): 待评估的已训练神经接收机模型。
        device (torch.device): 推理设备(CPU 或 CUDA)。
        scenario (Scenario): 物理层及信道配置对象(包含 FFT 大小、CP 长度、径数等)。
        snr (float / int): 当前测试的比特信噪比 Eb/N0 (dB)。
        frames (int): 本次测试仿真的 OFDM 符号块数量。
        seed (int): 随机数种子(用于保证数据与噪声生成的可复现性)。

    返回:
        dict: 包含测试元数据与性能指标的单行字典，核心字段包括：
            - 'scenario', 'ebn0_db', 'seed', 'frames', 'total_bits': 实验配置与数据量；
            - '{method}_errors', '{method}_ber': 各方法(perfect_zf, ls_zf, ls_mmse, nn)的总误比特数与误码率；
            - 'difference_sum', 'difference_square_sum': NN 与 LS+ZF 逐块 BER 配对差值的一阶和与二阶和(供后续置信区间计算);
            - 'zf_mse', 'mmse_mse': 传统均衡符号的平均均方误差。
    """
    features, labels, errors, mse = make_data(
        scenario,
        snr,
        frames,
        seed,
    )

    predicted = []

    with torch.inference_mode():
        for batch in torch.from_numpy(features).split(4096):
            logits = model(batch.to(device))
            predicted.append(
                logits.argmax(dim=1).cpu().numpy()
            )

    predicted = np.concatenate(predicted)

    # 将类别还原为bit，再按OFDM块统计错误
    truth_bits = np.column_stack(
        [labels //2, labels % 2]
    ).reshape(frames,-1)

    predicted_bits = np.column_stack(
        [predicted // 2, predicted % 2]
    ).reshape(frames,-1)

    # axis = 1的意思是按照列比较，输出是一个1行frames列的数组。
    errors["nn"] = np.count_nonzero(
        truth_bits != predicted_bits,
        axis = 1,
    )

    bits_per_block = truth_bits.shape[1]
    row = {
        "scenario": scenario.name,
        "ebn0_db": snr,
        "seed": seed,
        "frames": frames,
        "total_bits": truth_bits.size,
    }

    for key in METHODS:
        row[key + "_errors"] = int(errors[key].sum())
        row[key + "_ber"] = (
            row[key + "_errors"] / truth_bits.size
        )
    # 只求一个difference是因为mmse的errors和zf的一样
    difference = (
        errors["nn"] - errors["ls_zf"]
    ) / bits_per_block

    row["difference_sum"] = float(difference.sum())
    row["difference_square_sum"] = float(
        np.square(difference).sum()
    )

    row["zf_mse"] = float(mse["zf_mse"].mean())
    row["mmse_mse"] = float(mse["mmse_mse"].mean())

    return row

def summarize(raw_rows):
    """按场景和信噪比合并多次随机种子的测试结果，生成最终统计汇总表。

    功能:
        1. 将 raw_rows 按 (scenario, ebn0_db) 分组，合并不同测试随机种子的样本量；
        2. 计算各算法(ZF / MMSE / NN)在合并样本上的总体 BER 与加权平均 MSE;
        3. 利用一阶和与二阶平方和重构全局样本方差，估算 NN 与 LS+ZF 块级配对 ΔBER 的 95% 置信区间。

    参数:
        raw_rows (list[dict]): 由 evaluate_trial 返回的逐次测试结果字典列表。

    返回:
        list[dict]: 每个 (scenario, ebn0_db) 组合对应一条汇总字典，包含总帧数、总比特数、
            各方法总体误码率 ('{method}_ber')、配对差值 ('delta_ber')、95% 置信区间
            ('ci_low', 'ci_high') 及平均均方误差 ('zf_mse', 'mmse_mse')。
    """
    groups = {}

    for row in raw_rows:
        key = (row["scenario"], row["ebn0_db"])
        groups.setdefault(key,[]).append(row)

    summary = []

    for (name, snr), rows in groups.items():
        blocks = sum(r["frames"] for r in rows)
        total_bits = sum(r["total_bits"] for r in rows)

        result = {
            "scenario": name,
            "ebn0_db": snr,
            "frames": blocks,
            "total_bits": total_bits,
        }

        for key in METHODS:
            errors = sum(
                r[key + "_errors"] for r in rows
            )
            result[key + "_errors"] = errors
            result[key + "_ber"] = errors / total_bits

        difference = (
            sum(r["difference_sum"] for r in rows) / blocks
        )

        square_sum = sum(
            r["difference_square_sum"] for r in rows
        )

        varience = max(
            (
                square_sum
                - blocks * difference ** 2
            ) / (blocks - 1),
            0.0,
        )

        half_width = 1.96 * math.sqrt(
            varience / blocks
        )

        result["delta_ber"] = difference
        result["ci_low"] = difference - half_width
        result["ci_high"] = difference + half_width

        for key in ("zf_mse", "mmse_mse"):
            result[key] = sum(
                r[key] * r["frames"] for r in rows
                ) / blocks

        summary.append(result)

    return summary

# 关于这个函数的一些解释写在2026/11/14纸质版上面了
def save_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)   

# 画三类图，用于比较NN与传统方法的差异以及体现MMSE与ZF在MSE上的区别
def plot_results(output, scenarios, summary):
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    # 每个场景单独生成一张ber图
    for scenario in scenarios:
        rows = [
            r for r in summary
            if r["scenario"] == scenario.name
        ]

        snrs = [r["ebn0_db"] for r in rows]

        fig,ax = plt.subplots(figsize=(7,5))
        # 给每种场景的每一种均衡方式都创造一个存放ber的数组
        for key, (label,style) in METHODS.items():
            values = np.array(
                [r[key + "_ber"] for r in rows]
            )

            ax.semilogy(
                snrs,
                np.where(values > 0, values, np.nan),
                style,
                label = label,
            )

        ax.set(
            xlabel = "Data-bit Eb/N0 (dB)",
            ylabel = "BER",
            title = scenario.name,
        )
        ax.grid(True, which="both")
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            output / f"ber_{scenario.name}.png",
            dpi=180,
        )
        plt.close(fig)

    # 画最高SNR下不同场景的误差棒图像
    highest_snr = max(
        r["ebn0_db"] for r in summary
    )
    rows = [
        r for r in summary
        if r["ebn0_db"] == highest_snr
    ]

    delta = np.array(
        [r["delta_ber"] for r in rows]
    )
    width = np.array([
        (r["ci_high"] - r["ci_low"]) / 2
        for r in rows
    ])

    fig, ax = plt.subplots(figsize = (8,5))

    ax.errorbar(
        delta,
        np.arange(len(rows)),
        xerr = width,
        fmt="o",
    )
    ax.set_yticks(
        np.arange(len(rows)),
        [r["scenario"] for r in rows],
    )
    ax.axvline(
        0,
        color="gray",
        linestyle = "--",
    )
    ax.set(
        xlabel="BER(NN) - BER(LS+ZF); approximate 95% interval",
        title=f"Fixed checkpoint, Eb/N0 = {highest_snr} dB",
    )
    ax.grid(True, axis="x")

    fig.tight_layout()
    fig.savefig(
        output / "comparison.png",
        dpi = 180,
    )
    plt.close(fig)
    # 当前 QPSK 下 ZF/MMSE 的 BER 会重合，另用 MSE 对比
    rows = [
        r for r in summary
        if r["scenario"] == "baseline"
    ]

    fig, ax = plt.subplots(figsize=(7,5))

    for key,label in [
        ("zf_mse","LS+ZF"),
        ("mmse_mse","LS+MMSE"),
    ]:
        ax.semilogy(
            [r["ebn0_db"] for r in rows],
            [r[key] for r in rows],
            "-o",
            label = label,
        )

    ax.set(
        xlabel = "Data-bit Eb/N0 (dB)",
        ylabel = "Symbol MSE",
    )
    ax.grid(True, which = "both")
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output / "mse_baseline.png",
        dpi=180,
    )
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required = True,
        type=Path,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke","full"],
        default="smoke",
    )
    parser.add_argument(
        "--frames",
        type = int,
        default = 1000,
    )
    parser.add_argument(
        "--repeats",
        type = int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261109,
    )

    args = parser.parse_args()
    if args.frames < 2 or args.repeats < 1 or args.seed < 0:
        raise ValueError(
            "frames >= 2, repeats >= 1, seed >= 0 required"
        )

    if args.mode == "smoke":
        frames = 20
        repeats = 1
        snrs =[0,12,20]
    else:
        frames = args.frames
        repeats = args.repeats
        snrs = list(range(0,21,4))

    torch.set_num_threads(
        min(4,torch.get_num_threads())
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = args.checkpoint.resolve()

    # 还未搭建框架，只是把所需要的参数搬移到CPU中
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    # week3.train代码第247行
    if checkpoint["simulation"] != simulation_signature():
        raise ValueError(
            "week 1/2 settings changed since this checkpoint"
        )

    if checkpoint["config"]["mode"] != "normal":
        raise ValueError(
            "use a normal-training checkpoint(model)"
        )
    
    # 搭建模型框架
    model = NeuralReceiver(
        checkpoint["config"]["hidden_dim"]
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )
    model.eval()

    scenarios = make_scenarios(
        checkpoint["simulation"]
    )

    stamp = datetime.now().strftime(
        "%d%m%Y_%H%M%S_%f"
    )
    output = ROOT / "runs" / f"week4_{args.mode}_{stamp}"
    # 允许上一级目录不存在，但是不允许这个路径已经存在，防止覆盖
    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    metadata = {
        "mode":args.mode,
        "checkpoint":str(checkpoint_path),
        "checkpoint_sha256":hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_epoch":checkpoint["epoch"],
        "training_config":checkpoint["config"],
        "training_simulation":checkpoint["simulation"],
        "scenarios":[
            asdict(s) for s in scenarios
        ],
        "frames_per_trial":frames,
        "repeats":repeats,
        "snrs":snrs,
        "seed":args.seed,
        "python":sys.version,
        "torch":torch.__version__,
        "numpy":np.__version__,
        "device":str(device),
    }

    (output / "config.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Device",device,flush=True)
    print("Output",output,flush=True)

    raw = []

    for scenario in scenarios:
        # enumerate会同时创造序号和信噪比，比如1对应0，2对应4等
        for index, snr in enumerate(snrs):
            for repeat in range(repeats):
                seed = int(
                    np.random.SeedSequence(
                        # 这里不把scenario放到种子里是因为想让不同场景但是同一个
                        # repeat和snr下使用的是同一个种子，更加公平
                        [args.seed, index, repeat]
                    ).generate_state(1)[0]
                )

                row = evaluate_trial(
                    model,
                    device,
                    scenario,
                    snr,
                    frames,
                    seed,
                )
                row["repeat"] = repeat

                raw.append(row)

                save_csv(
                    output / "raw.csv",
                    raw,
                )

                print(
                    f"{scenario.name:12s} | "
                    f"{snr:2d} dB | "
                    f"repeat {repeat + 1}/{repeats} | "
                    f"ZF={row['ls_zf_ber']:.6f} | "
                    f"NN={row['nn_ber']:.6f}",
                    flush = True,
                )
    summary = summarize(raw)

    save_csv(
        output / "summary.csv",
        summary,
    )
    plot_results(
        output,
        scenarios,
        summary,
    )

    print(
        "Finished:",
        flush = True,
    )

if __name__ == "__main__":
    main()

# 这里的flat8表现差主要是因为丢弃的三条信道功率太大
# 可以证明，这回等效于导致信道估计的随机误差
# 而信道估计的误差又会导致均衡的象限偏移，导致误码