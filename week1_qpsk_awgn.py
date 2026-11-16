import math

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def qpsk_modulate(bits):
    """
    Modulate a sequence of bits using QPSK
    """
    bits = np.asarray(bits,dtype=np.int8)

    if bits.ndim != 1 or bits.size % 2 != 0 or bits.size == 0:
        raise ValueError("bits 必须是一维数组，且长度为偶数且不为零")

    bit_pairs = bits.reshape(-1,2)

    real_part = 2*bit_pairs[:,0]-1
    image_part = 2*bit_pairs[:,1]-1

    symbols = (real_part + 1j*image_part) / math.sqrt(2)

    return symbols

def qpsk_demodulate(received_symbols):
    """
    Demodulate a sequence of QPSK symbols into bits
    """
    received_symbols = np.asarray(received_symbols,dtype = np.complex128)
    received_symbols = received_symbols.reshape(-1)

    bits = np.empty(received_symbols.size * 2,dtype=np.int8)

    bits[0::2]= (received_symbols.real > 0).astype(np.int8)
    bits[1::2]= (received_symbols.imag > 0).astype(np.int8)

    return bits

def add_awgn_noise(symbols,ebn0_db,rng):
    """
    Add AWGN noise to a sequence of symbols
    """
    ebn0_linear = 10**(ebn0_db/10)
    noise_std = np.sqrt(1/(4*ebn0_linear))

    noise = noise_std*(rng.standard_normal(symbols.shape)+1j*rng.standard_normal(symbols.shape))

    return symbols + noise

def calculate_ber(sent_bits, detected_bits):
    """
    Calculate the bit error rate (BER) between sent and detected bits
    """
    sent_bits = np.asarray(sent_bits)
    detected_bits = np.asarray(detected_bits)

    if sent_bits.shape != detected_bits.shape:
        raise ValueError("发送比特和检测比特的形状必须相同")

    error_count = np.count_nonzero(sent_bits != detected_bits)

    ber = error_count / sent_bits.size

    return ber

def theoretical_qpsk_ber(ebn0_values_db):
    """
    Calculate the theoretical BER for QPSK over AWGN channel
    """
    ebn0_values_db = np.asarray(ebn0_values_db)

    ebn0_values_linear = 10**(ebn0_values_db/10)

    ber_values = [0.5 * (math.erfc(math.sqrt(value)))for value in ebn0_values_linear ]

    return np.array(ber_values)

def simulate_ber(ebn0_values_db,num_bits,seed=2026):
    """
    Simulate the BER for QPSK over AWGN channel
    """

    if num_bits % 2 != 0 or num_bits <= 0:
        raise ValueError("num_bits 必须为正偶数")

    rng = np.random.default_rng(seed)
    ber_values =[]

    for ebn0_db in ebn0_values_db:
        sent_bits= rng.integers(0,2,num_bits,dtype = np.int8)

        transmitted_symbols = qpsk_modulate(sent_bits)
        received_symbols = add_awgn_noise(transmitted_symbols,ebn0_db,rng)
        detected_bits = qpsk_demodulate(received_symbols)
        ber = calculate_ber(sent_bits,detected_bits)
        ber_values.append(ber)

    return np.asarray(ber_values)

def main():
    rng = np.random.default_rng(2026)
    test_bits = rng.integers(0,2,20,np.int8)

    test_symbols = qpsk_modulate(test_bits)
    recovered_bits = qpsk_demodulate(test_symbols)

    print("无噪声误码数：",np.count_nonzero(test_bits != recovered_bits))
    print("平均符号能量：",f"{np.mean(np.abs(test_symbols)**2):.4f}")

    constellation_bits = rng.integers(0,2,4000,np.int8)
    constellation_symbols = qpsk_modulate(constellation_bits)
    received_symbols = add_awgn_noise(constellation_symbols,4,rng)
    
    plt.figure(figsize=(6,6))
    plt.scatter(
        received_symbols.real,
        received_symbols.imag,
        s=8,
        alpha=0.35,
        label="接收符号"
    )

    ideal_symbols = np.asarray([1+1j, 1-1j, -1+1j, -1-1j])/np.sqrt(2)
    plt.scatter(
        ideal_symbols.real,
        ideal_symbols.imag,
        s=80,
        color="red",
        marker="x",
        label="理想符号"
    )
    plt.xlabel("实部")
    plt.ylabel("虚部")
    plt.title("QPSK 星座图 (Eb/N0 = 4 dB)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.savefig("qpsk_constellation.png", dpi=300)
    plt.show()

    ebn0_values_db = np.arange(-2,10,2)

    simulated_ber = simulate_ber(ebn0_values_db,100000,seed=2026)
    theoretical_ber = theoretical_qpsk_ber(ebn0_values_db)

    print("信噪比 (Eb/N0) [dB]:",ebn0_values_db)
    print("仿真BER:",simulated_ber)
    print("理论BER:",theoretical_ber)

    plt.figure(figsize=(8,5))
    plt.semilogy(
        ebn0_values_db,
        simulated_ber,
        'o-',
        label="仿真BER"
    )
    plt.semilogy(
        ebn0_values_db,
        theoretical_ber,
        's-',
        label="理论BER"
    )
    plt.xlabel("信噪比 (Eb/N0) [dB]")
    plt.ylabel("比特误码率 (BER)")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("qpsk_ber1.png", dpi=300)

    plt.show()

if __name__ == "__main__":
    main()