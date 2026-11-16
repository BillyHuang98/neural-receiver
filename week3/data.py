import numpy as np
import torch
from torch.utils.data import Dataset

import week2_ofdm as ofdm
from week1_qpsk_awgn import qpsk_modulate,qpsk_demodulate

FEATURE_NAMES = ["y_real","y_imag","h_ls_real","h_ls_imag"]

def simulation_signature():
    """记录通信参数和星座映射，供加载模型时核对"""

    constellation = qpsk_modulate(
        np.array([0,0,0,1,1,0,1,1],dtype = np.int8)
    )

    return {
        "n_fft": int(ofdm.N_FFT),
        "n_cp": int(ofdm.N_CP),
        "n_taps": int(ofdm.N_TAPS),
        "pilot_steps": int(ofdm.PILOT_STEP),

        "pilot_symbol": [
            float(np.real(ofdm.PILOT_SYMBOL)),
            float(np.imag(ofdm.PILOT_SYMBOL))
        ],

        "qpsk_mapping": np.column_stack([constellation.real,constellation.imag]).tolist(),
        "features": FEATURE_NAMES,
    }

class OFDMSymbolDataset(Dataset):
    def __init__(self,num_frames,ebn0_range,seed):
        if num_frames < 1:
            raise ValueError("num_frames must be positive")

        low,high = ebn0_range

        if low > high:
            raise ValueError("Invalid SNR range")
        rng = np.random.default_rng(seed)

        symbols_per_frame = ofdm.DATA_INDEX.size
        sample_count = symbols_per_frame * num_frames

        features = np.empty((sample_count,4),dtype = np.float32)
        labels = np.empty(sample_count,dtype = np.int64)

        error_counts ={
            "LS + ZF": 0,
            "LS + MMSE": 0,
        }
        for frame_id in range(num_frames):
            # 每一帧随机选择一个SNR
            ebn0_db = rng.uniform(low,high)

            noise_var = 1 / (2 * 10 ** (ebn0_db / 10))

            bits,_,_,transmitted = ofdm.make_frame(rng)

            received,_ = ofdm.multipath_channel(
                transmitted,
                noise_var,
                rng,
            )

            received_freq = ofdm.ofdm_demodulate(received)
            h_ls = ofdm.estimate_channel_ls(received_freq)

            y = received_freq[ofdm.DATA_INDEX]
            h = h_ls[ofdm.DATA_INDEX]
            # 这里的start到stop是一帧的符号数，即48
            start = frame_id * symbols_per_frame
            stop = start + symbols_per_frame

            features[start:stop] = np.column_stack(
                [y.real, y.imag, h.real, h.imag]
            )
            # 将比特对转换成两列的矩阵
            pairs = bits.reshape(-1,2)

            # 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3
            labels[start:stop] = (
                2 * pairs[:,0] + pairs[:,1]
            )

            # 计算传统接收机的BER
            estimates = {
                "LS + ZF": ofdm.equalize_zf(received_freq,h_ls)[ofdm.DATA_INDEX],
                "LS + MMSE": ofdm.equalize_mmse(received_freq,h_ls,noise_var)[ofdm.DATA_INDEX],
            }

            for name, symbols in estimates.items():
                predicted_bits = qpsk_demodulate(symbols)

                error_counts[name] += np.count_nonzero(
                    bits != predicted_bits
                )

        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

        self.baseline_ber = {
            name: float(errors / (2 * sample_count))
            for name, errors in error_counts.items()
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    




