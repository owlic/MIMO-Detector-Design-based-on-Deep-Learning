import tensorflow as tf
from generate_data import *


def piecewise_linear_soft_sign(x, t):
    Psi = -1 + tf.nn.relu(x + t) / tf.abs(t) - tf.nn.relu(x - t) / tf.abs(t)
    return Psi


def map_layer(x, input_size, output_size, foo=None):
    W0 = tf.Variable(tf.random.normal([input_size, output_size], stddev=0.01, seed=foo))
    b0 = tf.Variable(tf.random.normal([1, output_size], stddev=0.01, seed=foo))
    return tf.matmul(x, W0) + b0


def ReLU_layer(x, input_size, output_size, foo=None):
    W1 = tf.Variable(tf.random.normal([input_size, output_size], stddev=0.01, seed=foo))
    b1 = tf.Variable(tf.random.normal([1, output_size], stddev=0.01, seed=foo))
    return tf.nn.relu(tf.matmul(x, W1) + b1)


def PLSS_layer(x, input_size, output_size, t, foo=None):
    W2 = tf.Variable(tf.random.normal([input_size, output_size], stddev=0.01, seed=foo))
    b2 = tf.Variable(tf.random.normal([1, output_size], stddev=0.01, seed=foo))
    return piecewise_linear_soft_sign(tf.matmul(x, W2) + b2, t)


def tanh_layer(x, input_size, output_size, foo=None):
    W3 = tf.Variable(tf.random.normal([input_size, output_size], stddev=0.01, seed=foo))
    b3 = tf.Variable(tf.random.normal([1, output_size], stddev=0.01, seed=foo))
    return tf.math.tanh(tf.matmul(x, W3) + b3)


class Model:
    def __init__(self):
        self.B = C.batch_size
        self.K = C.K
        self.N = C.N
        self.seed_outset = C.seed_outset
        self.seed_record = [[] for i in range(3)]

        self.x = tf.compat.v1.placeholder(tf.float32, shape=[self.B, self.K])
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[self.B, self.N])
        self.H = tf.compat.v1.placeholder(tf.float32, shape=[self.B, self.N, self.K])
        self.N0 = tf.compat.v1.placeholder(tf.float32, shape=[self.B, 1])

        self.HT_y = tf.squeeze(tf.matmul(tf.linalg.matrix_transpose(self.H),
                                         tf.expand_dims(self.y, 2)), axis=2)   # shape=[B, K]
        self.HT_H = tf.matmul(tf.linalg.matrix_transpose(self.H), self.H)      # shape=[B, K, K]
        self.N0I = tf.squeeze(tf.tensordot(self.N0, tf.eye(self.K), axes=0), axis=1)

    def linear_equalizer(self):
        xe_ZF = tf.squeeze(tf.matmul(tf.expand_dims(self.HT_y, 1),
                                     tf.linalg.inv(self.HT_H)), axis=1)
        xe_MMSE = tf.squeeze(tf.matmul(tf.expand_dims(self.HT_y, 1),
                                       tf.linalg.inv(self.HT_H + self.N0I)), axis=1)
        loss_ZF = tf.reduce_mean(tf.square(self.x - xe_ZF))
        loss_MMSE = tf.reduce_mean(tf.square(self.x - xe_MMSE))

        return xe_ZF, xe_MMSE, loss_ZF, loss_MMSE

    def DetNet(self, xe_ZF):
        xk = [tf.zeros([self.B, self.K])]
        vk = [tf.zeros([self.B, C.v])]
        loss = [tf.zeros([])]

        for i in range(1, C.L + 1):
            seed_select = [self.seed_outset[0][0] + i,
                           self.seed_outset[0][1] + i,
                           self.seed_outset[0][2] + i]
            if i == 1 or i == C.L:
                self.seed_record[0].append(seed_select)
            HT_H_xk = tf.squeeze(tf.matmul(tf.expand_dims(xk[-1], 1), self.HT_H), 1)
            Z = tf.concat([self.HT_y, vk[-1], xk[-1], HT_H_xk], 1)
            zk = ReLU_layer(Z, 3 * self.K + C.v, C.HL, seed_select[0])
            xk.append(PLSS_layer(zk, C.HL, self.K, C.t, seed_select[1]))
            xk[i] = (1 - C.Res_rate) * xk[i] + C.Res_rate * xk[i - 1]   # k = 0, ..., L
            vk.append(map_layer(zk, C.HL, C.v, seed_select[2]))
            vk[i] = (1 - C.Res_rate) * vk[i] + C.Res_rate * vk[i - 1]
            Temp_loss = tf.reduce_mean(tf.reduce_mean(tf.square(self.x - xk[-1]), 1) /
                                       tf.reduce_mean(tf.square(self.x - xe_ZF), 1))
            loss.append(tf.math.log(float(i)) * Temp_loss)              # Base e
        return xk, loss

    def DetNet2(self, xe_ZF):
        xk = [tf.zeros([self.B, self.K])]
        loss = [tf.zeros([])]

        for i in range(1, C.L + 1):
            seed_select = [self.seed_outset[0][0] + i,
                           self.seed_outset[0][1] + i]
            if i == 1 or i == C.L:
                self.seed_record[0].append(seed_select)
            HT_H_xk = tf.squeeze(tf.matmul(tf.expand_dims(xk[-1], 1), self.HT_H), 1)
            Z = tf.concat([self.HT_y, xk[-1], HT_H_xk], 1)
            if C.method == '(ReLU)':
                zk = ReLU_layer(Z, 3 * self.K, C.HL, seed_select[0])
            elif C.method == '(PLSS)':
                zk = PLSS_layer(Z, 3 * self.K, C.HL, C.t, seed_select[0])
            elif C.method == '(tanh)':
                zk = tanh_layer(Z, 3 * self.K, C.HL, seed_select[0])
            else:
                raise BaseException("Method Not Support")
            xk.append(map_layer(zk, C.HL, self.K, seed_select[1]))
            xk[i] = (1 - C.Res_rate) * xk[i] + C.Res_rate * xk[i - 1]
            Temp_loss = tf.reduce_mean(tf.reduce_mean(tf.square(self.x - xk[-1]), 1) /
                                       tf.reduce_mean(tf.square(self.x - xe_ZF), 1))
            loss.append(tf.math.log(float(i)) * Temp_loss)
        return xk, loss

    def train(self, Iter, train_step):
        seed_select = self.seed_outset[1] + Iter
        if Iter == 1 or Iter == C.train_Iter:
            self.seed_record[1].append(seed_select)
        x_bit_train, x_train, y_train, H_train, N0_train = \
            generate_normal_data(C.modulation, C.batch_size, C.K, C.N,
                                 C.SNR_min, C.SNR_max, C.Gray_encode, seed_select)
        train_step.run(feed_dict={self.x: x_train, self.y: y_train, self.H: H_train, self.N0: N0_train})
