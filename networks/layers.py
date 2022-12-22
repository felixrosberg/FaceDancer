import math

import numpy as np
import scipy.signal as sps
import scipy.special as spspec
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv2D, Dense, Embedding,
                                     Flatten, Input, InputSpec, Layer,
                                     LeakyReLU, MaxPooling2D, Multiply, ReLU,
                                     Reshape, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import conv_utils

tf.keras.utils.disable_interactive_logging()

class AdaIN(Layer):
    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = Dense(self.x_channels)
        self.dense_2 = Dense(self.x_channels)

    def call(self, inputs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))
        return ys * x + yb

    def get_config(self):
        config = {
            #'w_channels': self.w_channels,
            #'x_channels': self.x_channels
        }
        base_config = super(AdaIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaIN1D(Layer):
    def __init__(self, **kwargs):
        super(AdaIN1D, self).__init__(**kwargs)

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = Dense(self.x_channels)
        self.dense_2 = Dense(self.x_channels)

    def call(self, inputs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, self.x_channels))
        return ys * x + yb

    def get_config(self):
        config = {
            #'w_channels': self.w_channels,
            #'x_channels': self.x_channels
        }
        base_config = super(AdaIN1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Conv2DMod(Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 demod=True,
                 **kwargs):
        super(Conv2DMod, self).__init__(**kwargs)
        self.filters = filters
        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.demod = demod
        self.input_spec = [InputSpec(ndim = 4),
                            InputSpec(ndim = 2)]

    def build(self, input_shape):
        channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        if input_shape[1][-1] != input_dim:
            raise ValueError('The last dimension of modulation input should be equal to input dimension.')

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Set input spec.
        self.input_spec = [InputSpec(ndim=4, axes={channel_axis: input_dim}),
                            InputSpec(ndim=2)]
        self.built = True

    def call(self, inputs):

        #To channels last
        x = tf.transpose(inputs[0], [0, 3, 1, 2])

        #Get weight and bias modulations
        #Make sure w's shape is compatible with self.kernel
        w = K.expand_dims(K.expand_dims(K.expand_dims(inputs[1], axis = 1), axis = 1), axis = -1)

        #Add minibatch layer to weights
        wo = K.expand_dims(self.kernel, axis = 0)

        #Modulate
        weights = wo * (w+1)

        #Demodulate
        if self.demod:
            d = K.sqrt(K.sum(K.square(weights), axis=[1,2,3], keepdims = True) + 1e-8)
            weights = weights / d

        #Reshape/scale input
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]]) # Fused => reshape minibatch to convolution groups.
        w = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])

        x = tf.nn.conv2d(x, w,
                strides=self.strides,
                padding="SAME",
                data_format="NCHW")

        # Reshape/scale output.
        x = tf.reshape(x, [-1, self.filters, x.shape[2], x.shape[3]]) # Fused => reshape convolution groups back to minibatch.
        x = tf.transpose(x, [0, 2, 3, 1])

        return x

    def compute_output_shape(self, input_shape):
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'demod': self.demod
        }
        base_config = super(Conv2DMod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AdaptiveAttention(Layer):

    def __init__(self, **kwargs):
        super(AdaptiveAttention, self).__init__(**kwargs)

    def call(self, inputs):
        m, a, i = inputs
        return (1 - m) * a + m * i

    def get_config(self):
        base_config = super(AdaptiveAttention, self).get_config()
        return base_config


def aad_block(inputs, c_out):
    h, z_att, z_id = inputs

    h_norm = BatchNormalization()(h)
    h = Conv2D(filters=c_out, kernel_size=1, kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(h_norm)

    m = Activation('sigmoid')(h)

    z_att_gamma = Conv2D(filters=c_out,
                         kernel_size=1,
                         kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(z_att)

    z_att_beta = Conv2D(filters=c_out,
                        kernel_size=1,
                        kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(z_att)

    a = Multiply()([h_norm, z_att_gamma])
    a = Add()([a, z_att_beta])

    z_id_gamma = Dense(h_norm.shape[-1],
                       kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(z_id)
    z_id_gamma = Reshape(target_shape=(1, 1, h_norm.shape[-1]))(z_id_gamma)

    z_id_beta = Dense(h_norm.shape[-1],
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(z_id)
    z_id_beta = Reshape(target_shape=(1, 1, h_norm.shape[-1]))(z_id_beta)

    i = Multiply()([h_norm, z_id_gamma])
    i = Add()([i, z_id_beta])

    h_out = AdaptiveAttention()([m, a, i])

    return h_out


def aad_block_mod(inputs, c_out):
    h, z_att, z_id = inputs

    h_norm = BatchNormalization()(h)
    h = Conv2D(filters=c_out, kernel_size=1, kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(h_norm)

    m = Activation('sigmoid')(h)

    z_att_gamma = Conv2D(filters=c_out,
                         kernel_size=1,
                         kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(z_att)

    z_att_beta = Conv2D(filters=c_out,
                        kernel_size=1,
                        kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(z_att)

    a = Multiply()([h_norm, z_att_gamma])
    a = Add()([a, z_att_beta])

    z_id_gamma = Dense(h_norm.shape[-1],
                       kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(z_id)

    i = Conv2DMod(filters=c_out,
                  kernel_size=1,
                  padding='same',
                  kernel_initializer='he_uniform',
                  kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))([h_norm, z_id_gamma])

    h_out = AdaptiveAttention()([m, a, i])

    return h_out


def aad_res_block(inputs, c_in, c_out):
    h, z_att, z_id = inputs

    if c_in == c_out:
        aad = aad_block([h, z_att, z_id], c_out)
        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(act)

        aad = aad_block([conv, z_att, z_id], c_out)
        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(act)

        h_out = Add()([h, conv])
        return h_out
    else:
        aad = aad_block([h, z_att, z_id], c_in)
        act = ReLU()(aad)
        h_res = Conv2D(filters=c_out,
                       kernel_size=3,
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(act)

        aad = aad_block([h, z_att, z_id], c_in)
        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(act)

        aad = aad_block([conv, z_att, z_id], c_out)
        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.001))(act)

        h_out = Add()([h_res, conv])

        return h_out


def aad_res_block_mod(inputs, c_in, c_out):
    h, z_att, z_id = inputs

    if c_in == c_out:
        aad = aad_block_mod([h, z_att, z_id], c_out)

        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(act)

        aad = aad_block_mod([conv, z_att, z_id], c_out)
        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(act)

        h_out = Add()([h, conv])
        return h_out
    else:
        aad = aad_block_mod([h, z_att, z_id], c_in)
        act = ReLU()(aad)
        h_res = Conv2D(filters=c_out,
                       kernel_size=3,
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(act)

        aad = aad_block_mod([h, z_att, z_id], c_in)
        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(act)

        aad = aad_block_mod([conv, z_att, z_id], c_out)
        act = ReLU()(aad)
        conv = Conv2D(filters=c_out,
                      kernel_size=3,
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l1(l1=0.0001))(act)

        h_out = Add()([h_res, conv])

        return h_out

