from keras.layers import *
from keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


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

    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)

    x = Add()([x, r])

    return x


def residual_down_block_small(inputs, filters, resample=True):
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


def get_discriminator():
    x_target = Input(shape=(256, 256, 3))

    x_0 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x_target)  # 256

    x_1 = residual_down_block(x_0, 128)

    x_2 = residual_down_block(x_1, 256)

    x_3 = residual_down_block(x_2, 512)

    x_4 = residual_down_block(x_3, 512)

    x_5 = residual_down_block(x_4, 512)

    x_6 = residual_down_block(x_5, 512)

    x = Conv2D(filters=512, kernel_size=4, strides=1)(x_6)  # 256
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=1, strides=1)(x)  # 256
    x = Reshape(target_shape=(1,))(x)

    dis_model = Model(x_target, x)
    dis_model.summary()

    return dis_model

