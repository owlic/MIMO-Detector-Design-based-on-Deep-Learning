import numpy as np
import time


def data_converter(Num, base, size, power):
    x = []
    while Num > 0:
        x.insert(0, Num % base)
        Num = Num // base
    for i in range(size - len(x)):
        x.insert(0, 0)
    return power * (np.array(x) * 2 - base + 1)


# WARNING! This function can only be used by QPSK.
def ML_detect(x, y, H, x_base):
    B = H.shape[0]
    K = H.shape[2]
    complexity = x_base ** K
    x_power = np.unique(np.reshape(x, [np.size(x)]))[int(x_base / 2)]
    xe = np.zeros([B, K])
    time_start = time.time()

    for i in range(B):
        distance_min = complexity
        for j in range(complexity):
            x_try = data_converter(j, x_base, K, x_power)
            distance = np.mean(np.square(y[i] - H[i].dot(x_try)))
            if distance < distance_min:
                xe[i] = x_try
                distance_min = distance

    time_cost = np.around(((time.time() - time_start) / 60), decimals=2)

    BER = np.mean(np.not_equal(x, xe).astype(float))

    return BER, time_cost
