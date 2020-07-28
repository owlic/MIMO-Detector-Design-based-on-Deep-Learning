import sys
from models import *
from maximum_likelihood import ML_detect


os.environ['KMP_WARNINGS'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "No GPUs found."

sess = tf.compat.v1.InteractiveSession()
# tf.compat.v1.disable_eager_execution()


class Record:
    def __init__(self, TextName):
        self.out_file = open(TextName, 'a')
        self.old_stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.old_stdout.write(text)
        self.out_file.write(text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout


def decode(signal, bits):
    if bits == 2:
        return np.where(signal <= 0., 0, 1)
    elif bits == 4:
        if C.Gray_encode:
            codeword = {3: [1, 0], 1: [1, 1], -1: [0, 1], -3: [0, 0]}
        else:
            codeword = {3: [1, 1], 1: [1, 0], -1: [0, 1], -3: [0, 0]}
    elif bits == 6:
        if C.Gray_encode:
            codeword = {7:  [1, 0, 0], 5:  [1, 0, 1], 3:  [1, 1, 1], 1:  [1, 1, 0],
                        -7: [0, 0, 0], -5: [0, 0, 1], -3: [0, 1, 1], -1: [0, 1, 0]}
        else:
            codeword = {7:  [1, 1, 1], 5:  [1, 1, 0], 3:  [1, 0, 1], 1:  [1, 0, 0],
                        -7: [0, 0, 0], -5: [0, 0, 1], -3: [0, 1, 0], -1: [0, 1, 1]}
    else:
        raise BaseException("Bits number error")

    B = signal.shape[0]
    K = signal.shape[1]
    signal_decode = [[] for n in range(B)]

    for p in range(B):
        for q in range(K):
            signal_decode[p].append(codeword[signal[p][q]])

    signal_decode = np.reshape(signal_decode, [B, int(K / 2 * bits)], order='F')

    return signal_decode


def contrast(x_bit_target, x_target, x_hat, bits):
    x_base = int(2 ** (bits / 2))
    x_sort = np.unique(np.reshape(x_target, [np.size(x_target)])).astype(np.float32)
    assert x_sort.size == x_base, "Data x error"

    boundary = np.zeros(x_sort.size)
    Loc_revised = np.zeros(x_sort.size)

    for n in range(x_sort.size):
        if n == x_sort.size - 1:
            boundary[n] = 50.
        else:
            boundary[n] = (x_sort[n] + x_sort[n + 1]) / 2   # <class 'numpy.float64'>
        Loc_revised[n] = (100. - x_sort.size + 1) + n * 2

    x_hat_edited = x_hat

    for n in range(len(boundary)):
        x_hat_edited = np.where(x_hat_edited < boundary[n], Loc_revised[n], x_hat_edited)

    x_hat_bit_edited = x_hat_edited

    for n in range(len(Loc_revised)):
        x_hat_bit_edited = np.where(x_hat_bit_edited == Loc_revised[n],
                                    int(Loc_revised[n] - 100), x_hat_bit_edited)

    x_hat_bit = decode(x_hat_bit_edited, bits)   # <class 'numpy.int32'>
    BER = np.mean(np.not_equal(x_bit_target.astype(np.int32), x_hat_bit).astype(float))

    # x_hat_sym_edited = x_hat_edited
    #
    # for n in range(len(Loc_revised)):
    #     x_hat_sym_edited = np.where(x_hat_sym_edited == Loc_revised[n],
    #                                 x_sort[n], x_hat_sym_edited)
    #
    # x_source_sym = np.reshape(x_source.astype(np.float32), [C.batch_size, C.Nt, 2], order='F')
    # x_hat_sym = np.reshape(x_hat_sym_edited, [C.batch_size, C.Nt, 2], order='F')
    # SER = np.mean(np.logical_not(np.all(x_source_sym == x_hat_sym, axis=2)).astype(float))

    return BER


def test(Iter):
    SNRdB = np.linspace(C.SNRdB_min, C.SNRdB_max, C.Num_SNR)
    SNR = 10 ** (SNRdB / 10)
    BER_ZF = np.zeros([C.Num_SNR, C.test_count])
    BER_MMSE = np.zeros([C.Num_SNR, C.test_count])
    BER_DL = np.zeros([C.Num_SNR, C.test_count])
    BER_ML = np.full([C.Num_SNR], np.nan)

    for p in range(C.Num_SNR):
        for q in range(C.test_count):
            test_seed = C.seed_outset[2] + q + 1
            if not model.seed_record[2] or (p + q == 0) or (p + q == C.Num_SNR + C.test_count - 2):
                model.seed_record[2].append(test_seed)
            x_bit_test, x_test, y_test, H_test, N0_test = \
                generate_test_set(C.modulation, C.batch_size, C.K, C.N, SNR[p], SNR[p], C.Gray_encode, test_seed)
            xe_ZF_test, xe_MMSE_test, xk_test = \
                sess.run([xe_ZF, xe_MMSE, xk],
                         {model.x: x_test, model.y: y_test, model.H: H_test, model.N0: N0_test})
            BER_ZF[p][q] = contrast(x_bit_test, x_test, xe_ZF_test, C.bits[C.modulation])
            BER_MMSE[p][q] = contrast(x_bit_test, x_test, xe_MMSE_test, C.bits[C.modulation])
            BER_DL[p][q] = contrast(x_bit_test, x_test, xk_test[-1], C.bits[C.modulation])
            if C.do_ML and Iter == C.train_Iter and q == C.test_count - 1:
                BER_ML[p], _ = ML_detect(x_test, y_test, H_test, C.bits[C.modulation])

    BER_Avg_ZF = np.mean(BER_ZF, axis=1)
    BER_Avg_MMSE = np.mean(BER_MMSE, axis=1)
    BER_Avg_DL = np.mean(BER_DL, axis=1)

    improve_rate = np.divide(np.array(BER_Avg_MMSE), np.array(BER_Avg_DL))
    total_improve_rate = np.mean(improve_rate)

    with Record(record_file):
        print("\ntesting time(hr): %.3f" % ((time.time() - time_start) / 3600))
        print("Average BER(ZF):\n", BER_Avg_ZF)
        print("Average BER(MMSE):\n", BER_Avg_MMSE)
        print("Average BER(DL):\n", BER_Avg_DL)
        if C.do_ML and Iter == C.train_Iter:
            print("BER(ML):\n", BER_ML)

        print("improve rate (vs MMSE):\n", improve_rate)
        print(f'total improve rate (vs MMSE): {total_improve_rate:.3%}\n')

    return SNRdB, BER_Avg_ZF, BER_Avg_MMSE, BER_Avg_DL, BER_ML


def auto_limit(*result, ceil=1e-00, floor=1e-06):
    all_result = []
    for n in result:
        all_result += [n]
    all_result = np.sort(np.array(all_result)[~ np.isnan(all_result)])
    if all_result[0] != 0 and all_result[-1] != 0:
        ceil = 10 ** (np.ceil(np.log10(all_result[-1])))
        floor = 10 ** (np.floor(np.log10(all_result[0])))
    else:
        pass
    return ceil, floor


def plot(count, SNRdB, BER_Avg_ZF, BER_Avg_MMSE, BER_Avg_DL, BER_ML):
    Chart = plt.figure()
    ax = Chart.add_subplot(111)
    ax.yaxis.set_ticks_position('both')
    top, bottom = auto_limit(BER_Avg_ZF, BER_Avg_MMSE, BER_Avg_DL, BER_ML)
    plt.xlim(SNRdB[0], SNRdB[-1])
    plt.ylim(bottom, top)
    plt.xticks(SNRdB)
    plt.yscale('log')
    plt.grid(True)
    plt.plot(SNRdB, BER_Avg_ZF, linestyle='-', label='ZF')
    plt.plot(SNRdB, BER_Avg_MMSE, marker='x', linestyle='-', label='MMSE')
    plt.plot(SNRdB, BER_Avg_DL, marker='o', linestyle='-', label=C.algorithm)
    if not np.isnan(BER_ML[0]):
        plt.plot(SNRdB, BER_ML, marker='P', linestyle='-', label='ML')
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.legend(loc='best')
    plt.title(f'{C.modulation}: {C.Nt}x{C.Nr}')
    plt.savefig(f'{C.algorithm}｜{C.modulation}｜{C.Nt}x{C.Nr}｜Gray={C.Gray_encode}｜'
                f'SNR(dB)={C.SNRdB_min}-{C.SNRdB_max}｜L={C.L}｜HL={C.HL}｜v={C.v}｜'
                f't={C.t}｜RR={C.Res_rate}｜B={C.batch_size}｜Iter={count}.png')


model = Model()
xe_ZF, xe_MMSE, loss_ZF, loss_MMSE = model.linear_equalizer()
if C.modulation == 'QPSK':
    xk, loss_DL = model.DetNet(xe_ZF)
elif C.modulation == '16QAM' or C.modulation == '64QAM':
    xk, loss_DL = model.DetNet2(xe_ZF)
else:
    raise BaseException("Modulation Not Support")

total_loss = tf.add_n(loss_DL)

global_step = tf.Variable(0, trainable=False)
lr = tf.compat.v1.train.exponential_decay(
    C.learning_rate, global_step, C.decay_steps, C.decay_rate, C.staircase)
train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
train_writer = tf.compat.v1.summary.FileWriter(f'logs/L={C.L}_HL={C.HL}_v={C.v}', sess.graph)
Init = tf.compat.v1.global_variables_initializer()
sess.run(Init)

record_file = f'{C.algorithm}｜{C.modulation}｜{C.Nt}x{C.Nr}｜Gray={C.Gray_encode}｜' \
              f'SNR(dB)={C.SNRdB_min}-{C.SNRdB_max}｜L={C.L}｜HL={C.HL}｜' \
              f'v={C.v}｜t={C.t}｜RR={C.Res_rate}｜B={C.batch_size}｜Iter={C.train_Iter}｜' \
              f'Seed={model.seed_record[0][0]}-{C.seed_outset[1]}-{C.seed_outset[2]}.txt'


# Training model
loss_train = []
BER_train = []
time_cost = []
fixed_info = []
data_path = f'ValidationSet/SNR={C.SNR_min:.1f}-{C.SNR_max:.1f}/' \
            f'{C.modulation}｜K={C.K}｜N={C.N}｜B={C.batch_size}｜Gary={C.Gray_encode}'
# VLD: validate
x_bit_VLD = np.loadtxt(f'{data_path}/B_x_bit.txt').reshape(
    [C.batch_size, C.Nt * C.bits[C.modulation]])
x_VLD = np.loadtxt(f'{data_path}/B_x.txt').reshape([C.batch_size, C.K])
y_VLD = np.loadtxt(f'{data_path}/B_y.txt').reshape([C.batch_size, C.N])
H_VLD = np.loadtxt(f'{data_path}/B_H.txt').reshape([C.batch_size, C.N, C.K])
N0_VLD = np.loadtxt(f'{data_path}/B_N0.txt').reshape([C.batch_size, 1])
if C.do_ML:
    BER_ML_VLD, TimeCost = ML_detect(x_VLD, y_VLD, H_VLD, C.bits[C.modulation])
    with Record(record_file):
        print(f'BER(ML): {BER_ML_VLD:f}   time(min): {TimeCost}')

time_start = time.time()

for i in range(1, C.train_Iter + 1):
    model.train(i, train_step)
    # validating model
    if i % 1000 == 0:
        loss_ZF_VLD, loss_MMSE_VLD, loss_DL_VLD, xe_ZF_VLD, xe_MMSE_VLD, xk_VLD = \
            sess.run([loss_ZF, loss_MMSE, loss_DL[C.L], xe_ZF, xe_MMSE, xk],
                     {model.x: x_VLD, model.y: y_VLD, model.H: H_VLD, model.N0: N0_VLD})
        BER_ZF_VLD = contrast(x_bit_VLD, x_VLD, xe_ZF_VLD, C.bits[C.modulation])
        BER_MMSE_VLD = contrast(x_bit_VLD, x_VLD, xe_MMSE_VLD, C.bits[C.modulation])
        BER_DL_VLD = contrast(x_bit_VLD, x_VLD, xk_VLD[-1], C.bits[C.modulation])
        if not loss_train:
            fixed_info = [loss_ZF_VLD, loss_MMSE_VLD, BER_ZF_VLD, BER_MMSE_VLD]
            with Record(record_file):
                print("-------------------------------------------------")
                print("| loss(ZF)    loss(MMSE)  BER(ZF)     BER(MMSE) |")
                print(f'| {loss_ZF_VLD:<f}    {loss_MMSE_VLD:<f}  '
                      f'  {BER_ZF_VLD:<f}    {BER_MMSE_VLD:<f}  |')
                print("-------------------------------------------------")
                print("  iter｜ loss[L]     BER[L]      time(min)")
        else:
            assert fixed_info == \
                   [loss_ZF_VLD, loss_MMSE_VLD, BER_ZF_VLD, BER_MMSE_VLD], "Fixed set error"
        loss_train.append(np.around(loss_DL_VLD, decimals=6))
        BER_train.append(np.around(BER_DL_VLD, decimals=6))
        time_cost.append(np.around(((time.time() - time_start) / 60), decimals=3))
        with Record(record_file):
            print(f'{i:>6d}｜ {loss_DL_VLD:<f}    {BER_DL_VLD:<f}    {time_cost[-1]:<.2f}')
        time_start = time.time()
    # Testing model
    if i % 10000 == 0:
        SNRdB_test, BER_Avg_ZF_test, BER_Avg_MMSE_test, BER_Avg_DL_test, BER_ML_test = test(i)
        plot(i, SNRdB_test, BER_Avg_ZF_test, BER_Avg_MMSE_test, BER_Avg_DL_test, BER_ML_test)
        time_start = time.time()


with Record(record_file):
    print("total training time(hr): %.3f" % (np.sum(time_cost) / 60))
    print("\n--------------------------------------------------------------------------------\n")
    print("loss of training:\n", loss_train, "\n")
    print("BER of training:\n", BER_train)
    print("\n--------------------------------------------------------------------------------\n")
    print("seed select (network):", model.seed_record[0])
    print("seed select (train):", model.seed_record[1])
    print("seed select (test):", model.seed_record[2])

saver = tf.compat.v1.train.Saver()
save_path = saver.save(
    sess, f'models/{C.algorithm}｜{C.modulation}｜{C.Nt}x{C.Nr}｜L={C.L}｜HL={C.HL}｜v={C.v}/'
          f'{C.algorithm}｜{C.modulation}｜{C.Nt}x{C.Nr}｜L={C.L}｜HL={C.HL}｜v={C.v}.ckpt')
