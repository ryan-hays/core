import tensorflow as tf
import numpy as np
import util

def weight_variable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.02))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv3d(x, W, stride=2):
    return tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')

def deconv3d(x, W, output_shape, stride=2):
    return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride, stride, 1], padding='SAME')

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

class Generator(object):

    def __init__(self, z_size, name="g_"):
        with tf.variable_scope(name):
            self.name = name

            self.W = {
                'h1': weight_variable([z_size, 2*2*2*128]),
                'h2': weight_variable([4, 4, 4, 64, 128]),
                'h3': weight_variable([4, 4, 4, 32, 64]),
                'h4': weight_variable([4, 4, 4, 16, 32]),
                'h5': weight_variable([4, 4, 4, 14, 16])
            }

            self.b = {
                'h5': bias_variable([1])
            }

    def __call__(self, z):
        shape = z.get_shape().as_list()

        h = tf.nn.relu(vbn(tf.matmul(z, self.W['h1']), 'g_vbn_1'))
        h = tf.reshape(h, [-1, 2, 2, 2, 128])
        h = tf.nn.relu(vbn(deconv3d(h, self.W['h2'], [shape[0], 4, 4, 4, 64]), 'g_vbn_2'))
        h = tf.nn.relu(vbn(deconv3d(h, self.W['h3'], [shape[0], 8, 8, 8, 32]), 'g_vbn_3'))
        h = tf.nn.relu(vbn(deconv3d(h, self.W['h4'], [shape[0], 16, 16, 16, 16]), 'g_vbn_4'))
        x = tf.nn.tanh(deconv3d(h, self.W['h5'], [shape[0], 32, 32, 32, 1]) + self.b['h5'])
        return x

class Discriminator(object):

    def __init__(self, name="d_"):
        with tf.variable_scope(name):
            self.name = name
            self.n_kernels = 300
            self.dim_per_kernel = 50

            self.W = {
                'h1': weight_variable([4, 4, 4, 14, 16]),
                'h2': weight_variable([4, 4, 4, 16, 32]),
                'h3': weight_variable([4, 4, 4, 32, 64]),
                'h4': weight_variable([4, 4, 4, 64, 128]),
                'h5': weight_variable([2*2*2*128+self.n_kernels, 2]),
                'md': weight_variable([2*2*2*128, self.n_kernels*self.dim_per_kernel])
            }

            self.b = {
                'h1': bias_variable([16]),
                'h5': bias_variable([2]),
                'md': bias_variable([self.n_kernels])
            }

            self.bn2 = BatchNormalization([32], 'bn2')
            self.bn3 = BatchNormalization([64], 'bn3')
            self.bn4 = BatchNormalization([128], 'bn4')

    def __call__(self, x, train):
        shape = x.get_shape().as_list()
        noisy_x = x + tf.random_normal([shape[0], 32, 32, 32, 1])

        h = lrelu(conv3d(noisy_x, self.W['h1']) + self.b['h1'])
        h = lrelu(self.bn2(conv3d(h, self.W['h2']), train))
        h = lrelu(self.bn3(conv3d(h, self.W['h3']), train))
        h = lrelu(self.bn4(conv3d(h, self.W['h4']), train))
        h = tf.reshape(h, [-1, 2*2*2*128])

        m = tf.matmul(h, self.W['md'])
        m = tf.reshape(m, [-1, self.n_kernels, self.dim_per_kernel])
        abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(m, 3) - tf.expand_dims(tf.transpose(m, [1, 2, 0]), 0)), 2)
        f = tf.reduce_sum(tf.exp(-abs_dif), 2) + self.b['md']

        h = tf.concat(1, [h, f])
        y = tf.matmul(h, self.W['h5']) + self.b['h5']
        return y        