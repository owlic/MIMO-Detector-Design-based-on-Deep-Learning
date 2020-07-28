

class Const(object):
    __slots__ = ()
    method = '(PLSS)'
    modulation = '16QAM'
    bits = {'QPSK': 2, '16QAM': 4, '64QAM': 6}
    Nt = 8
    Nr = 16
    K = Nt * 2                                   # real & image part
    N = Nr * 2                                   # real & image part
    Gray_encode = True
    SNRdB_min = 13
    SNRdB_max = 18
    SNR_min = 10 ** (SNRdB_min / 10)
    SNR_max = 10 ** (SNRdB_max / 10)
    Num_SNR = 6
    L = 90                                       # number of layers
    if modulation == 'QPSK':
        algorithm = 'DetNet'
        v = K                                    # size of v
        do_ML = True
        Gray_encode = False
    else:
        algorithm = 'DetNet2' + method
        v = None
        do_ML = False
    HL = 3 * K + int(v or 0)                     # size of hidden layer
    t = 0.03                                     # hyperparameter from PLSS
    batch_size = 10000
    train_Iter = 200000
    test_count = 100
    Res_rate = 0.2                               # from ResNet
    decay_steps = 1000
    decay_rate = 0.97
    staircase = True
    learning_rate = 0.0001                       # The initial learning rate
    seed_outset = [[0, L, 2 * L], 0, 1000000]    # network / training / testing
    bias = 0.00005                               # (recommended) 0.008% if 4(Nt)x8(Nr), 0.005% if 8(Nt)x16(Nr)


C = Const()
