import tensorflow as tf


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.applications import VGG16


def perceptual_backbone(feature_layers, input_shape):

    def backbone(x):
        extractor = VGG16(input_shape=input_shape, include_top=False, weights='imagenet')
        extractor.trainable = False

        feature_maps = [extractor.get_layer(layer_name).output for layer_name in feature_layers]

        out = Model(extractor.input, feature_maps)(x)

        return out

    return backbone


def perceptual_model(backbone, input_shape):
    inputs = Input(shape=input_shape)
    x = backbone(inputs)

    return Model(inputs, x)


def perceptual_loss(input_shape, blocks, block_weights):
    percept_model = perceptual_model(perceptual_backbone(blocks, input_shape), input_shape)

    def loss_function(y_true, y_pred, bw=block_weights, smooth=0.85, content_id=3):
        f_true = percept_model(y_true)
        f_pred = percept_model(y_pred)

        loss = 0
        tick = 0
        for feature_map_true, feature_map_pred in zip(f_true, f_pred):

            if tick == content_id:
                distance = tf.reduce_mean(tf.square(feature_map_true - feature_map_pred))
                loss += distance
            distance = tf.reduce_mean(tf.square(feature_map_true * smooth - feature_map_pred * smooth)) * bw[tick]
            loss += distance
            tick += 1

        return loss

    return loss_function


def perceptual_loss_flagged(input_shape, blocks, block_weights):
    percept_model = perceptual_model(perceptual_backbone(blocks, input_shape), input_shape)

    def loss_function(y_true, y_pred, flags, bw=block_weights, smooth=0.85):
        f_true = percept_model(y_true)
        f_pred = percept_model(y_pred)

        loss = 0
        tick = 0
        for feature_map_true, feature_map_pred in zip(f_true, f_pred):

            distance = tf.reduce_mean(tf.abs(feature_map_true * smooth - feature_map_pred * smooth),
                                      axis=[1, 2, 3]) * bw[tick]

            adjusted_loss = tf.reduce_sum(tf.multiply(distance, tf.cast(flags, tf.float32))) / (
                    tf.reduce_sum(tf.cast(flags, tf.float32)) + 0.001)

            loss += adjusted_loss
            tick += 1

        return loss

    return loss_function


@tf.function
def fs_reconstruction_loss(y_true, y_pred, flags):
    rec = tf.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)),
                         axis=[1, 2, 3])
    reconstruction_loss = tf.reduce_sum(tf.multiply(rec, tf.cast(flags, tf.float32))) / (
            tf.reduce_sum(tf.cast(flags, tf.float32)) + 0.001)
    reconstruction_loss = tf.clip_by_value(reconstruction_loss, clip_value_min=0, clip_value_max=100)

    return reconstruction_loss / 2


@tf.function
def fs_reconstruction_loss_l1(y_true, y_pred, flags):
    rec = tf.reduce_mean(tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)),
                         axis=[1, 2, 3])
    reconstruction_loss = tf.reduce_sum(tf.multiply(rec, tf.cast(flags, tf.float32))) / (
            tf.reduce_sum(tf.cast(flags, tf.float32)) + 0.001)
    reconstruction_loss = tf.clip_by_value(reconstruction_loss, clip_value_min=0, clip_value_max=100)

    return reconstruction_loss / 2


def perceptual_similarity_backbone(feature_layers, path):
    extractor = load_model(path).layers[1]
    extractor.trainable = False

    backbone = Model(extractor.input, [extractor.get_layer(layer_name).output for layer_name in feature_layers])

    return backbone


def perceptual_similarity_loss(blocks, block_weights, block_margin, path):
    percept_model = perceptual_similarity_backbone(blocks, path)

    def loss_function(y_true, y_pred, bw=block_weights, bm=block_margin):
        f_true = percept_model(y_true)
        f_pred = percept_model(y_pred)

        loss = 0
        tick = 0
        bs = f_true[0].shape[0]
        for feature_map_true, feature_map_pred in zip(f_true, f_pred):
            distance = tf.reduce_mean(1 + tf.losses.cosine_similarity(tf.reshape(feature_map_true, shape=(bs, -1)),
                                                                      tf.reshape(feature_map_pred, shape=(bs, -1))))
            d_loss = tf.nn.relu(distance - bm[tick]) * bw[tick]
            loss += d_loss
            tick += 1

        return loss

    return loss_function

