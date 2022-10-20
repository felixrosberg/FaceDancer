from keras.models import Model
from keras.layers import *
from tensorflow_addons.layers import InstanceNormalization
from networks.layers import AdaIN, AdaptiveAttention

import numpy as np


def residual_down_block(inputs, filters, resample=True):
    x = inputs

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = AveragePooling2D()(r)

    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = AveragePooling2D()(x)

    x = Add()([x, r])

    return x


def residual_up_block(inputs, filters, resample=True, name=None):
    x, z_id = inputs

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def adaptive_attention(inputs, filters, name=None):
    x_t, x_s = inputs

    m = Concatenate(axis=-1)([x_t, x_s])
    m = Conv2D(filters=filters // 4, kernel_size=3, strides=1, padding='same')(m)
    m = LeakyReLU(0.2)(m)
    m = InstanceNormalization()(m)
    m = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', activation='sigmoid', name=name)(m)

    x = AdaptiveAttention()([m, x_t, x_s])

    return x


def adaptive_attention_double(inputs, filters, name=None):
    x_t, x_s = inputs

    c = Concatenate(axis=-1)([x_t, x_s])

    m_hat = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(c)
    m_hat = LeakyReLU(0.2)(m_hat)
    m_hat = InstanceNormalization()(m_hat)
    m_hat = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', activation='sigmoid', name=name + '_hat')(m_hat)

    m = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(c)
    m = LeakyReLU(0.2)(m)
    m = InstanceNormalization()(m)
    m = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same', activation='sigmoid', name=name)(m)

    m_hat = m * m_hat

    x = AdaptiveAttention()([m_hat, x_t, x_s])

    return x


def adaptive_fusion_up_block(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = adaptive_attention([x_t, x_s], x_t.shape[-1], name=name)

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def adaptive_fusion_up_block_double(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = adaptive_attention_double([x_t, x_s], x_t.shape[-1], name=name)

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def dual_adaptive_fusion_up_block(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = adaptive_attention([x_t, x_s], x_t.shape[-1], name=name + '_0')
    x = adaptive_attention([x_t, x], x_t.shape[-1], name=name + '_1')

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


def adaptive_fusion_up_block_concat_baseline(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = Concatenate(axis=-1)([x_t, x_s])

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add(name=name if name == 'final' else None)([x, r])

    return x


def adaptive_fusion_up_block_add_baseline(inputs, filters, resample=True, name=None):
    x_t, x_s, z_id = inputs

    x = Add()([x_t, x_s])

    r = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    if resample:
        r = UpSampling2D(interpolation='bilinear')(r)

    x = InstanceNormalization()(x)
    x = AdaIN()([x, z_id])
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    if resample:
        x = UpSampling2D(interpolation='bilinear')(x)

    x = Add()([x, r])

    return x


# helper function for choosing feature fusion method
def make_layer(l_type, inputs, filters, resample, name=None):
    if l_type == 'affa':
        return adaptive_fusion_up_block(inputs, filters, resample=resample, name=name)
    if l_type == 'd_affa':
        return dual_adaptive_fusion_up_block(inputs, filters, resample=resample, name=name)
    if l_type == 'do_affa':
        return adaptive_fusion_up_block_double(inputs, filters, resample=resample, name=name)
    elif l_type == 'concat':
        return adaptive_fusion_up_block_concat_baseline(inputs, filters, resample=resample, name=name)
    elif l_type == 'add':
        return adaptive_fusion_up_block_add_baseline(inputs, filters, resample=resample, name=name)
    elif l_type == 'no_skip':
        return residual_up_block(inputs[1:], filters, resample=resample)


def get_generator(up_types=None, mapping_depth=4, mapping_size=256):

    # if up_types=None, use a default setting
    if up_types is None:
        up_types = ['no_skip', 'no_skip', 'affa', 'affa', 'affa', 'concat']

    x_target = Input(shape=(256, 256, 3))
    z_source = Input(shape=(512,))

    # build mapping network M
    z_id = z_source
    for m in range(np.max([mapping_depth - 1, 0])):
        z_id = Dense(mapping_size)(z_id)
        z_id = LeakyReLU(0.2)(z_id)
    if mapping_depth >= 1:
        z_id = Dense(mapping_size)(z_id)

    # build generator network G
    x_0 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x_target)            # 256

    x_1 = residual_down_block(x_0, 128)                                                     # 128

    x_2 = residual_down_block(x_1, 256)                                                     # 64

    x_3 = residual_down_block(x_2, 512)                                                     # 32

    x_4 = residual_down_block(x_3, 512)                                                     # 16

    x_5 = residual_down_block(x_4, 512)                                                     # 8

    x_6 = residual_down_block(x_5, 512, resample=False)                                     # 8

    u_5 = residual_up_block([x_6, z_id], 512, resample=False)                               # 8

    u_4 = make_layer(up_types[0], [x_5, u_5, z_id], 512, resample=True, name='16x16')       # 16

    u_3 = make_layer(up_types[1], [x_4, u_4, z_id], 512, resample=True, name='32x32')       # 32

    u_2 = make_layer(up_types[2], [x_3, u_3, z_id], 256, resample=True, name='64x64')       # 64

    u_1 = make_layer(up_types[3], [x_2, u_2, z_id], 128, resample=True, name='128x128')     # 128

    u_0 = make_layer(up_types[4], [x_1, u_1, z_id], 64, resample=True, name='256x256')      # 256

    out = make_layer(up_types[5], [x_0, u_0, z_id], 3, resample=False, name='final')        # 256

    gen_model = Model([x_target, z_source], out)
    gen_model.summary()

    return gen_model





