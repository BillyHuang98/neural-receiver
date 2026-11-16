from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from week1_qpsk_awgn import qpsk_modulate, qpsk_demodulate

N_FFT = 64
N_CP = 16
N_TAPS = 5
PILOT_STEP = 4
PILOT_SYMBOL = 1 + 0j

PILOT_INDEX = np.arange(0,N_FFT,PILOT_STEP)

DATA_INDEX = np.setdiff1d(np.arange(N_FFT),PILOT_INDEX)

def ofdm_modulate(freq_symbols):

    time_signal = np.fft.ifft(freq_symbols,norm='ortho')
    transmitted_signals = np.concatenate([time_signal[-N_CP:], time_signal])

    return transmitted_signals

def ofdm_demodulate(received_signal):

    useful_signal = received_signal[N_CP:N_CP+N_FFT]
    received_freq = np.fft.fft(useful_signal,norm = 'ortho')

    return received_freq

def make_frame(rng):

    bits = rng.integers(0,2,size = 2 * DATA_INDEX.size,dtype = np.int8)
    data_symbols = qpsk_modulate(bits)

    freq_symbols = np.zeros(N_FFT,dtype = np.complex128)

    freq_symbols[PILOT_INDEX] = PILOT_SYMBOL
    freq_symbols[DATA_INDEX] = data_symbols

    transmitted_signal = ofdm_modulate(freq_symbols)

    return bits,data_symbols,freq_symbols,transmitted_signal

def multipath_channel(transmitted_signal,noise_var,rng):

    powers = np.exp(-np.arange(N_TAPS))
    powers = powers / np.sum(powers)

    taps = np.sqrt(powers / 2) * (rng.standard_normal(N_TAPS) + 1j * rng.standard_normal(N_TAPS)) 

    received_signal = np.convolve(transmitted_signal,taps)

    noise = np.sqrt(noise_var / 2) * (rng.standard_normal(received_signal.size) + 1j * rng.standard_normal(received_signal.size))

    true_channel = np.fft.fft(taps,N_FFT)

    return received_signal + noise,true_channel

def estimate_channel_ls(received_freq):

    pilot_estimate = (received_freq[PILOT_INDEX]/PILOT_SYMBOL)

    taps_estimate = np.fft.ifft(pilot_estimate)

    taps_estimate = taps_estimate[ :N_TAPS]

    channel_estimate = np.fft.fft(taps_estimate,N_FFT)

    return channel_estimate

def equalize_zf(received_freq,channel_estimate):
    channel_power = np.abs(channel_estimate)**2

    equalized = (np.conj(channel_estimate) * received_freq / np.maximum(channel_power,1e-12))
    return equalized

def equalize_mmse(received_freq,channel_estimate,noise_var):
    channel_power = np.abs(channel_estimate)**2

    equalized = (np.conj(channel_estimate) * received_freq / (channel_power + noise_var))

    return equalized

def check_no_noise():
    assert N_CP >= N_TAPS - 1

    assert N_FFT % PILOT_STEP == 0
    assert N_TAPS <= PILOT_INDEX.size

    rng = np.random.default_rng(7)

    bits, _,freq_symbols,transmitted = make_frame(rng)

    received,true_channel = multipath_channel(transmitted,0.0,rng)

    received_freq = ofdm_demodulate(received)
    estimated_channel = estimate_channel_ls(received_freq)

    assert np.allclose(received_freq,true_channel*freq_symbols)
    assert np.allclose(estimated_channel,true_channel)

    equalized = equalize_zf(received_freq,estimated_channel)
    detected_bits = qpsk_demodulate(equalized[DATA_INDEX])

    errors = np.count_nonzero(bits != detected_bits)
    assert errors == 0

    print("No-noise check passed. Bit errors:",errors)

def simulate(ebn0_values_db,num_frames = 2000,seed = 2026):
    rng = np.random.default_rng(seed)

    ber = {
        "Perfect CSI + ZF":[],
        "LS + ZF":[],
        "LS + MMSE":[],
    }
    mse ={
        "LS + ZF":[],
        "LS + MMSE":[],
    }

    for ebn0_db in ebn0_values_db:
        noise_var = 1/(2*10**(ebn0_db/10))
        # 将ber中的每一个name对应的数组都赋值为0
        error_counts = {name:0 for name in ber}
        mse_sums = {name:0.0 for name in mse}

        for _ in range(num_frames):
            bits,data_symbols, _,transmitted = make_frame(rng)
            received, true_channel = multipath_channel(transmitted,noise_var,rng)
            received_freq = ofdm_demodulate(received)
            estimated_channel = estimate_channel_ls(received_freq)

            estimates = {
                "Perfect CSI + ZF": equalize_zf(received_freq,true_channel)[DATA_INDEX],

                "LS + ZF": equalize_zf(received_freq,estimated_channel)[DATA_INDEX],

                "LS + MMSE": equalize_mmse(received_freq,estimated_channel,noise_var)[DATA_INDEX],
            } 
            for name,estimated_symbols in estimates.items():
                detected_bits = qpsk_demodulate(estimated_symbols)

                error_counts[name] += np.count_nonzero(bits != detected_bits)

            for name in mse:
                mse_sums[name] += np.sum(
                    np.abs(estimates[name] - data_symbols) **2
                )
        total_symbols = num_frames * DATA_INDEX.size
        total_bits = 2 * total_symbols

        for name in ber:
            ber[name].append(
                error_counts[name]/total_bits
            )
        
        for name in mse:
            mse[name].append(
                mse_sums[name]/total_symbols
            )

        print(
            f"Eb/N0 = {ebn0_db:2d}dB |"
            f"Perfect = {ber['Perfect CSI + ZF'][-1]:.6f} |"
            f"LS+ZF = {ber['LS + ZF'][-1]:.6f} |"
            f"LS+MMSE = {ber['LS + MMSE'][-1]:.6f}"
        )

    return ber,mse

def main():
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    check_no_noise()

    ebn0_values_db = np.arange(0,21,4)

    ber,mse = simulate(
        ebn0_values_db,
        num_frames=2000,
        seed = 2026,
    )

    fig,axes = plt.subplots(1,2,figsize=(12,4.5))

    styles = ["k-o","b-s","r--x"]

    for (name,values),style in zip(ber.items(),styles):
        axes[0].semilogy(
            ebn0_values_db,
            values,
            style,
            label = name,
        )
    axes[0].set_title("OFDM bit error rate")
    axes[0].set_ylabel("BER")

    for name, values in mse.items():
        axes[1].semilogy(
            ebn0_values_db,
            values,
            marker = 's',
            label = name,
        )
    axes[1].set_title("Symbol MSE before hard decisions")
    axes[1].set_ylabel("MSE")

    for ax in axes:
        ax.set_xlabel("Data-bit Eb/N0 (dB)")
        ax.grid(True,which = "both")
        ax.legend()

    fig.tight_layout()

    output_path = Path(__file__).with_name("week_2_results_seed2026.png")

    fig.savefig(output_path,dpi = 200)
    plt.show()

if __name__ =="__main__":
    main()

