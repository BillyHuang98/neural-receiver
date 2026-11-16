from dataclasses import dataclass

import numpy as np

from week1_qpsk_awgn import qpsk_modulate, qpsk_demodulate
from week2_ofdm import equalize_zf, equalize_mmse

# dataclass用于简化类的声明，避免写__init__函数
@dataclass(frozen= True)
class Scenario:
    name: str
    true_taps: int = 5
    rx_taps: int = 5
    pilot_step: int = 4
    # profile指的是Power Delay Profile
    # 即每条多径符合什么样的功率衰减规律
    profile: str = "exponential"
    n_fft: int = 64
    n_cp: int = 16

    def validate(self):
        if not 1 <= self.true_taps <= self.n_fft:
            raise ValueError("Invalid true_taps")

        if not self.true_taps -1 <= self.n_cp <= self.n_fft:
            raise ValueError("CP is too short or too long")

        if not 2 <= self.pilot_step <= self.n_fft:
            raise ValueError("Invalid pilot_step")

        if self.n_fft % self.pilot_step:
            raise ValueError("pilot_step must divide n_fft")

        pilot_count = self.n_fft // self.pilot_step

        if not 1 <= self.rx_taps <= pilot_count:
            raise ValueError("Too few pilots for rx_taps")

        if self.true_taps > pilot_count:
            raise ValueError("Pilot Aliasing!")

        if self.profile not in ("exponential", "uniform"):
            raise ValueError("Undefined power delay profile")
        
"""
    单帧/多帧 OFDM 链路仿真：
    - 输入：场景参数 scenario、信噪比 ebn0_db、帧数 frames、随机种子 seed 及可选 noise_var。
    - 输出：
        1. features: (frames * N_data, 4) 频域接收符号与估计信道的实虚部，供 NN 测试；
        2. labels: (frames * N_data,) 对应的 QPSK 真实标签类别 (0~3)
        3. errors: 记录 perfect_zf, ls_zf, ls_mmse 逐块误比特数的字典；
        4. mse: 记录 zf_mse, mmse_mse 逐块符号均方误差的字典。
    """
def make_data(scenario, ebn0_db, frames, seed, noise_var=None):
    scenario.validate()

    if frames < 1:
        raise ValueError("frames must be positve")

    s = scenario
    rng = np.random.default_rng(seed)
    # np.arange(起始序号, 终止序号, 间隔)
    pilots = np.arange(0, s.n_fft, s.pilot_step)
    # 从第一个参数中去除第二个参数
    data = np.setdiff1d(np.arange(s.n_fft), pilots)
    symbols_per_block = data.size

    if s.profile == "exponential":
        powers = np.exp(-np.arange(s.true_taps))
    else:
        powers = np.ones(s.true_taps)

    powers = powers / powers.sum()

    if noise_var is None:
        noise_var = 1 / (2 * 10 ** (ebn0_db / 10))

    features = np.empty(
        (frames * symbols_per_block, 4),
        dtype=np.float32,
    )
    labels = np.empty(
        frames * symbols_per_block,
        dtype=np.int64,
    )

    errors = {
        key: np.zeros(frames,dtype=np.int64)
        for key in ("perfect_zf", "ls_zf", "ls_mmse")
    }

    mse = {
        key: np.empty(frames)
        for key in("zf_mse", "mmse_mse")
    }

    for block in range(frames):
        rng.uniform(ebn0_db, ebn0_db)

        bits = rng.integers(
            0,
            2,
            2 * symbols_per_block,
            dtype=np.int8,
        )
        symbols = qpsk_modulate(bits)

        x_freq = np.zeros(s.n_fft, dtype=np.complex128)
        x_freq[pilots] = 1 + 0j
        x_freq[data] = symbols

        x_time = np.fft.ifft(x_freq, norm="ortho")
        # 切片冒号前后分别表示开始位置和终止位置
        prefix = x_time[-s.n_cp:] if s.n_cp else x_time[:0]
        transmitted = np.concatenate([prefix, x_time])

        taps = np.sqrt(powers / 2) * (
            rng.standard_normal(s.true_taps)
            + 1j * rng.standard_normal(s.true_taps)
        )

        received = np.convolve(transmitted, taps)

        received += np.sqrt(noise_var / 2) * (
            rng.standard_normal(received.size)
            + 1j * rng.standard_normal(received.size)
        )

        y_freq = np.fft.fft(
            received[s.n_cp:s.n_cp + s.n_fft],
            norm = "ortho",
        )

        h_true = np.fft.fft(taps, s.n_fft)

        estimated_taps = np.fft.ifft(y_freq[pilots])
        h_ls = np.fft.fft(
            estimated_taps[:s.rx_taps],
            s.n_fft,
        )

        start = block * symbols_per_block
        stop = start + symbols_per_block

        y = y_freq[data]
        h = h_ls[data]

        features[start:stop] = np.column_stack(
            [y.real, y.imag, h.real, h.imag]
        )

        pairs = bits.reshape(-1,2)
        labels[start:stop] = 2 * pairs[:, 0] + pairs[:, 1]

        estimates = {
            "perfect_zf": equalize_zf(y_freq, h_true)[data],
            "ls_zf": equalize_zf(y_freq, h_ls)[data],
            "ls_mmse": equalize_mmse(
                y_freq,
                h_ls,
                noise_var,
            )[data],
        }

        for key, estimate in estimates.items():
            errors[key][block] = np.count_nonzero(
                qpsk_demodulate(estimate) != bits
            )

        # abs对数组中的每一个数取模，mean将这些模加起来并求均值
        mse["zf_mse"][block] = np.mean(
            np.abs(estimates["ls_zf"] - symbols) ** 2
        )
        mse["mmse_mse"][block] = np.mean(
        np.abs(estimates["ls_mmse"] - symbols) ** 2
        )

    return features, labels, errors, mse