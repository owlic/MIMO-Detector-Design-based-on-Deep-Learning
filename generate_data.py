import numpy as np
import time
import os
import matplotlib.pyplot as plt
from parameters import *


def plot_generated_Gaussian(data_final, FigureName, figure_path):
    data = data_final.flatten()
    SD = np.linspace(-5, 5, 1001)   # Standard deviation
    sample_count = []
    for i in SD:
        if i == SD[0]:
            sample_count.append(np.sum(np.where(data < i + 0.005, 1, 0)))
        elif i != SD[-1]:
            sample_count.append(np.sum(np.where(data < i + 0.005, 1, 0)) -
                                np.sum(np.where(data < i - 0.005, 1, 0)))
        else:
            sample_count.append(np.sum(np.where(data > i - 0.005, 1, 0)))
    PDF = np.array(sample_count) / data.size
    plt.xlim(-5, 5)
    plt.ylim(0, 0.005)
    plt.plot(SD, PDF)
    plt.savefig(f'{figure_path}/{FigureName}.png')


def Gray2Binary(RBC):
    BC = np.zeros(np.shape(RBC))
    BC_ = np.insert(BC, 0, 0., axis=2).tolist()
    for i in range(RBC.shape[0]):
        for j in range(RBC.shape[1]):
            for k in range(RBC.shape[2]):
                BC_[i][j][k + 1] = np.abs(BC_[i][j][k] - RBC[i][j][k])
            BC[i][j] = BC_[i][j][1:]
    BC = np.reshape(BC, [RBC.shape[0], RBC.shape[1] * RBC.shape[2]], order='F')
    return BC


def generate_normal_data(modulation, B, K, N, SNR_min, SNR_max, Gray=True, foo=None, Es=1):
    np.random.seed(foo)
    B_HR = np.sqrt(1 / 2) * np.random.randn(B, int(N / 2), int(K / 2))   # real part (rayleigh)
    B_HI = np.sqrt(1 / 2) * np.random.randn(B, int(N / 2), int(K / 2))   # image part (rayleigh)
    B_H = np.append(np.append(B_HR, - B_HI, axis=2), np.append(B_HI, B_HR, axis=2), axis=1)

    if modulation == 'QPSK':
        B_x_bit = np.random.randint(2, size=[B, K])   # 0/1
        B_x = np.sqrt(Es / K) * (B_x_bit * 2 - 1)
    elif modulation == '16QAM':
        B_x_bit = np.random.randint(2, size=[B, K * 2])
        if Gray:
            B_x_bit_converted = Gray2Binary(np.reshape(B_x_bit, [B, K, 2], order='F'))
            B_x = np.sqrt(Es / (K * 5)) * (B_x_bit_converted * 2 - 1)    # √ [Es / (Nt x 10)]
        else:
            B_x = np.sqrt(Es / (K * 5)) * (B_x_bit * 2 - 1)
        A = np.append(2 * np.eye(K), np.eye(K), axis=1)
        B_A = np.tensordot(np.ones(B), A, axes=0)
        B_x = np.squeeze(np.matmul(B_A, np.expand_dims(B_x, 2)), 2)
    elif modulation == '64QAM':
        B_x_bit = np.random.randint(2, size=[B, K * 3])
        if Gray:
            B_x_bit_converted = Gray2Binary(np.reshape(B_x_bit, [B, K, 3], order='F'))
            B_x = np.sqrt(Es / (K * 21)) * (B_x_bit_converted * 2 - 1)   # √ [Es / (Nt x 42)]
        else:
            B_x = np.sqrt(Es / (K * 21)) * (B_x_bit * 2 - 1)
        A = np.append(4 * np.eye(K), 2 * np.eye(K), axis=1)
        A = np.append(A, np.eye(K), axis=1)
        B_A = np.tensordot(np.ones(B), A, axes=0)
        B_x = np.squeeze(np.matmul(B_A, np.expand_dims(B_x, 2)), 2)
    else:
        raise BaseException("Modulation Not Support")

    B_w = np.sqrt(1 / 2) * np.random.randn(B, N)
    B_y = np.zeros([B, N])
    B_N0 = np.zeros([B, 1])

    B_SNR = np.random.uniform(low=SNR_min, high=SNR_max, size=B)

    for i in range(B):
        B_N0[i] = Es / B_SNR[i]
        H = B_H[i]
        x = B_x[i]
        w = B_w[i]
        y = H.dot(x) + np.sqrt(B_N0[i]) * w
        B_y[i] = y

    return B_x_bit, B_x, B_y, B_H, B_N0


def generate_symmetry_data(data_shape, FileName=None, save_path=None, bias=None, Gaussian=True):
    data_size = np.size(np.ndarray(data_shape))
    assert data_size > 0 and data_size % 1000 == 0, "Data size error"
    if Gaussian:
        data_qualified = []
        attempt_count = 0
        while len(data_qualified) < 100:
            attempt_count += 1
            print(f'\r{FileName} attempts: {attempt_count}', end='')
            data_original = np.random.randn(data_size)
            data_halved = np.delete(np.sort(data_original).reshape([2, int(data_size / 2)]), 1, 0).flatten()
            if data_halved[-1] <= 0 and data_halved[- int(round(data_size * (0.002 - bias)))] >= -0.005:
                data_qualified.append(data_halved)
        print("")
        data_edited = np.mean(data_qualified, axis=0)
        data_sym = np.random.permutation(np.append(data_edited, - data_edited)).reshape(data_shape)
        plot_generated_Gaussian(data_sym, FileName, save_path)
    else:
        ones = np.ones(int(data_size / 2))
        zeros = np.zeros(int(data_size / 2))
        data_sym = np.random.permutation(np.append(ones, zeros)).reshape(data_shape)
    return data_sym


def generate_validation_set(modulation, B, K, N, SNR_min, SNR_max, Gray=True, bias=None, Es=1):
    FileName = ["B_x_bit", "B_x", "B_y", "B_H", "B_w", "B_N0"]
    save_path = f'ValidationSet/SNR={SNR_min:.1f}-{SNR_max:.1f}/{modulation}｜K={K}｜N={N}｜B={B}｜Gary={Gray}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    time_start = time.time()

    B_HR = np.sqrt(1 / 2) * generate_symmetry_data([B, int(N / 2), int(K / 2)], "B_HR", save_path, bias)
    B_HI = np.sqrt(1 / 2) * generate_symmetry_data([B, int(N / 2), int(K / 2)], "B_HI", save_path, bias)
    B_H = np.append(np.append(B_HR, - B_HI, axis=2), np.append(B_HI, B_HR, axis=2), axis=1)

    if modulation == 'QPSK':
        B_x_bit = generate_symmetry_data([B, K], Gaussian=False)
        B_x = np.sqrt(Es / K) * (B_x_bit * 2 - 1)
    elif modulation == '16QAM':
        B_x_bit = generate_symmetry_data([B, K * 2], Gaussian=False)
        if Gray:
            B_x_bit_converted = Gray2Binary(np.reshape(B_x_bit, [B, K, 2], order='F'))
            B_x = np.sqrt(Es / (K * 5)) * (B_x_bit_converted * 2 - 1)
        else:
            B_x = np.sqrt(Es / (K * 5)) * (B_x_bit * 2 - 1)
        A = np.append(2 * np.eye(K), np.eye(K), axis=1)
        B_A = np.tensordot(np.ones(B), A, axes=0)
        B_x = np.squeeze(np.matmul(B_A, np.expand_dims(B_x, 2)), 2)
    elif modulation == '64QAM':
        B_x_bit = generate_symmetry_data([B, K * 3], Gaussian=False)
        if Gray:
            B_x_bit_converted = Gray2Binary(np.reshape(B_x_bit, [B, K, 3], order='F'))
            B_x = np.sqrt(Es / (K * 21)) * (B_x_bit_converted * 2 - 1)
        else:
            B_x = np.sqrt(Es / (K * 21)) * (B_x_bit * 2 - 1)
        A = np.append(4 * np.eye(K), 2 * np.eye(K), axis=1)
        A = np.append(A, np.eye(K), axis=1)
        B_A = np.tensordot(np.ones(B), A, axes=0)
        B_x = np.squeeze(np.matmul(B_A, np.expand_dims(B_x, 2)), 2)
    else:
        raise BaseException("Modulation Not Support")

    B_w = np.sqrt(1 / 2) * generate_symmetry_data([B, N], "B_w", save_path, bias)

    time_cost = np.around(((time.time() - time_start) / 60), decimals=2)
    print("time cost (min) - generate validation set:", time_cost)

    B_y = np.zeros([B, N])
    B_N0 = np.zeros([B, 1])

    B_SNR = np.linspace(SNR_min, SNR_max, B)

    for i in range(B):
        B_N0[i] = Es / B_SNR[i]
        H = B_H[i]
        x = B_x[i]
        w = B_w[i]
        y = H.dot(x) + np.sqrt(B_N0[i]) * w
        B_y[i] = y

    data = {0: B_x_bit, 1: B_x, 2: B_y, 3: B_H, 4: B_w, 5: B_N0}

    for j in range(6):
        batch_count = 0
        with open(f'{save_path}/{FileName[j]}.txt', mode='w') as file:
            file.write(f'# Array shape: {format(data[j].shape)}\n')
            for data_slice in data[j]:
                batch_count += 1
                file.write(f'# {batch_count}\n')
                np.savetxt(file, data_slice, fmt='%9.6f')


def generate_test_set(modulation, B, K, N, SNR_min, SNR_max, Gray=True, foo=None):
    return generate_normal_data(modulation, B, K, N, SNR_min, SNR_max, Gray, foo)


if __name__ == '__main__':
    generate_validation_set(C.modulation, C.batch_size, C.K, C.N, C.SNR_min, C.SNR_max, C.Gray_encode, C.bias)
