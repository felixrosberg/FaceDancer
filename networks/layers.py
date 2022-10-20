import tensorflow as tf
import numpy as np
import scipy.signal as sps
import scipy.special as spspec

import tensorflow.keras.backend as K
import math
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Layer, InputSpec, Conv2D, LeakyReLU, Dense, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import ReLU, Activation, UpSampling2D, Add, Reshape, Multiply, Embedding
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras import initializers, constraints, regularizers
from tensorflow.keras.models import Model


def sin_activation(x, omega=30):
    return tf.math.sin(omega * x)


class MetricLayer(Layer):
    def __init__(self):
        super(MetricLayer, self).__init__()

    def call(self, x):
        total_codebook_metric_mae = tf.reduce_mean(tf.abs(x[0] - x[1]))
        total_codebook_metric_cos = tf.reduce_mean(1 + tf.losses.cosine_similarity(x[0], x[1]))

        self.add_metric(total_codebook_metric_mae, name='total_vector_mae')
        self.add_metric(total_codebook_metric_cos, name='total_vector_cos')
        return x[0]

    def get_config(self):
        config = {
        }
        base_config = super(MetricLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


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


class CreatePatches(tf.keras.layers.Layer ):

    def __init__( self , patch_size):
        super( CreatePatches , self).__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        patches = []
        # For square images only ( as inputs.shape[ 1 ] = inputs.shape[ 2 ] )
        input_image_size = inputs.shape[ 1 ]
        for i in range( 0 , input_image_size , self.patch_size ):
            for j in range( 0 , input_image_size , self.patch_size ):
                patches.append( inputs[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
        return patches

    def get_config(self):
        config = {'patch_size': self.patch_size,
        }
        base_config = super(CreatePatches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SelfAttention(tf.keras.layers.Layer ):

    def __init__( self , alpha, filters=128):
        super(SelfAttention , self).__init__()
        self.alpha = alpha
        self.filters = filters

        self.f = Conv2D(filters, 1, 1)
        self.g = Conv2D(filters, 1, 1)
        self.s = Conv2D(filters, 1, 1)

    def call(self, inputs):

        f_map = self.f(inputs)
        f_map = tf.image.transpose(f_map)

        g_map = self.g(inputs)

        s_map = self.s(inputs)

        att = f_map * g_map

        att = att / self.alpha

        return tf.keras.activations.softmax(att + s_map, axis=0)

    def get_config(self):
        config = {'alpha': self.alpha,
        }
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def get_config(self):
        config = {
        }
        base_config = super(Sampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = {
            'patch_size': self.patch_size,
        }
        base_config = super(Patches, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = {
            'num_patches': self.num_patches,
        }
        base_config = super(PatchEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Slicer(Layer):
    def __init__(self, emb_dim=512, **kwargs):
        super(Slicer, self).__init__(**kwargs)
        self.emb_dim = emb_dim

    def call(self, x):
        input_shape = tf.shape(x)
        x_0 = tf.slice(x, [0, 0], [input_shape[0], self.emb_dim // 2])
        x_1 = tf.slice(x, [0, self.emb_dim // 2], [input_shape[0], self.emb_dim // 2])
        return [x_0, x_1]

    def get_config(self):
        config = {
            'emb_dim': self.emb_dim,
        }
        base_config = super(Slicer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VectorQuantizer(Layer):
    def __init__(self, num_embeddings=40960, embedding_dim=512, beta=0.25, tag='0', **kwargs):
        super().__init__(**kwargs)
        self.tag = tag
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae_" + self.tag,
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        codebook_metric = tf.reduce_mean(tf.abs(quantized - tf.stop_gradient(x)))

        self.add_metric(codebook_metric, name='vector_mae')

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):



        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
                tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                + tf.reduce_sum(self.embeddings ** 2, axis=0)
                - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self):
        config = {
            'embedding_dim': self.embedding_dim,
            'num_embeddings': self.num_embeddings,
            'beta': self.beta,
        }
        base_config = super(VectorQuantizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VectorQuantizerSt(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        self.add_metric(commitment_loss, name='commitment_loss')
        self.add_metric(codebook_loss, name='codebook_loss')

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class VectorQuantizerCosineID(Layer):
    def __init__(self, num_embeddings=40960, embedding_dim=512, beta=0.5, min_arg=True, id_init=None, tag='0', **kwargs):
        super().__init__(**kwargs)
        self.tag = tag
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.min_arg = min_arg
        self.arg_scaler = -1 if min_arg else 1
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ) if id_init is None else id_init,
            trainable=True,
            name="embeddings_vqvae_" + self.tag,
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = self.beta * tf.reduce_mean(
            1 + tf.losses.cosine_similarity(tf.stop_gradient(quantized), x)
        )
        codebook_loss = tf.reduce_mean(1 + tf.losses.cosine_similarity(quantized, tf.stop_gradient(x)))
        self.add_loss(commitment_loss)
        self.add_loss(codebook_loss)

        codebook_metric = tf.reduce_mean(1 + tf.losses.cosine_similarity(quantized, x))

        self.add_metric(codebook_metric, name='vector_cos')

        self.add_metric(commitment_loss, name='commitment_loss')
        self.add_metric(codebook_loss, name='codebook_loss')

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        # calculate cosine similarity between feature vector and codes

        features_norm = tf.math.l2_normalize(flattened_inputs, axis=1)
        codes_norm = tf.math.l2_normalize(self.embeddings, axis=0)

        similarity = tf.matmul(features_norm, codes_norm)

        # Derive the indices for minimum distances.
        min_encoding_indices = tf.argmin(self.arg_scaler * similarity, axis=1)
        return min_encoding_indices

    def get_config(self):
        config = {
            'embedding_dim': self.embedding_dim,
            'num_embeddings': self.num_embeddings,
            'beta': self.beta,
        }
        base_config = super(VectorQuantizerCosineID, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, num_classes, margin=0.7, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists

    def get_config(self):
        config = {'num_classes': self.num_classes,
                  'margin': self.margin,
                  'logist_scale': self.logist_scale
        }
        base_config = super(ArcMarginPenaltyLogists, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class KLLossLayer(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""
    def __init__(self, beta=1.5, **kwargs):
        super(KLLossLayer, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputs):
        z_mean, z_log_var = inputs

        kl_loss = tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = -0.5 * kl_loss * self.beta

        self.add_loss(kl_loss * 0)
        self.add_metric(kl_loss, 'kl_loss')

        return inputs

    def get_config(self):
        config = {
            'beta': self.beta
        }
        base_config = super(KLLossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    def get_config(self):
        config = {
            'padding': self.padding,
        }
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ResBlock(Layer):

    def __init__(self, fil, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.fil = fil

        self.conv_0 = Conv2D(kernel_size=3, filters=fil, strides=1)
        self.conv_1 = Conv2D(kernel_size=3, filters=fil, strides=1)

        self.res = Conv2D(kernel_size=1, filters=1, strides=1)

        self.lrelu = LeakyReLU(0.2)
        self.padding = ReflectionPadding2D(padding=(1, 1))

    def call(self, inputs):
        res = self.res(inputs)

        x = self.padding(inputs)
        x = self.conv_0(x)
        x = self.lrelu(x)

        x = self.padding(x)
        x = self.conv_1(x)
        x = self.lrelu(x)

        out = x + res

        return out

    def get_config(self):
        config = {
            'fil': self.fil
        }
        base_config = super(ResBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SubpixelConv2D(Layer):
    """ Subpixel Conv2D Layer
    upsampling a layer from (h, w, c) to (h*r, w*r, c/(r*r)),
    where r is the scaling factor, default to 4
    # Arguments
    upsampling_factor: the scaling factor
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        the second and the third dimension increased by a factor of
        `upsampling_factor`; the last layer decreased by a factor of
        `upsampling_factor^2`.
    # References
        Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network Shi et Al. https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upsampling_factor=4, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                             'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space( inputs, self.upsampling_factor )

    def get_config(self):
        config = { 'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],
                 input_shape_1,
                 input_shape_2,
                 int(input_shape[3]/factor)
               ]
        return tuple( dims )


def id_mod_res(inputs, c):
    feature_map, z_id = inputs

    x = Conv2D(c, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(feature_map)

    x = AdaIN()([x, z_id])

    x = ReLU()(x)

    x = Conv2D(c, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)

    x = AdaIN()([x, z_id])

    out = Add()([x, feature_map])

    return out


def id_mod_res_v2(inputs, c):
    feature_map, z_id = inputs

    affine = Dense(feature_map.shape[-1])(z_id)
    x = Conv2DMod(c, kernel_size=3, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))([feature_map, affine])

    x = ReLU()(x)

    affine = Dense(x.shape[-1])(z_id)
    x = Conv2DMod(c, kernel_size=3, padding='same',
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))([x, affine])
    out = Add()([x, feature_map])

    x = ReLU()(x)

    return out


def simswap(im_size, filter_scale=1, deep=True):
    inputs = Input(shape=(im_size, im_size, 3))
    z_id = Input(shape=(512,))

    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = Conv2D(filters=64 // filter_scale, kernel_size=7, padding='valid', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)           # 112
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = Conv2D(filters=64 // filter_scale, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)  # 56
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = Conv2D(filters=256 // filter_scale, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)  # 28
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = Conv2D(filters=512 // filter_scale, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)  # 14
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)                         # 14

    if deep:
        x = Conv2D(filters=512 // filter_scale, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(x)  # 7
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)

    x = id_mod_res([x, z_id], 512 // filter_scale)

    x = id_mod_res([x, z_id], 512 // filter_scale)

    x = id_mod_res([x, z_id], 512 // filter_scale)

    x = id_mod_res([x, z_id], 512 // filter_scale)

    if deep:
        x = SubpixelConv2D(upsampling_factor=2)(x)
        x = Conv2D(filters=512 // filter_scale, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)

    x = SubpixelConv2D(upsampling_factor=2)(x)
    x = Conv2D(filters=256 // filter_scale, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = SubpixelConv2D(upsampling_factor=2)(x)
    x = Conv2D(filters=128 // filter_scale, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)                         # 56

    x = SubpixelConv2D(upsampling_factor=2)(x)
    x = Conv2D(filters=64 // filter_scale, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)                       # 112

    x = ReflectionPadding2D(padding=(3, 3))(x)
    out = Conv2D(filters=3, kernel_size=7, padding='valid')(x)            # 112

    model = Model([inputs, z_id], out)
    model.summary()

    return model


def simswap_v2(deep=True):
    inputs = Input(shape=(224, 224, 3))
    z_id = Input(shape=(512,))

    x = ReflectionPadding2D(padding=(3, 3))(inputs)
    x = Conv2D(filters=64, kernel_size=7, padding='valid', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)           # 112
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)  # 56
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)  # 28
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)  # 14
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)                         # 14

    if deep:
        x = Conv2D(filters=512, kernel_size=3, strides=2, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)  # 7
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)

    x = id_mod_res_v2([x, z_id], 512)

    x = id_mod_res_v2([x, z_id], 512)

    x = id_mod_res_v2([x, z_id], 512)

    x = id_mod_res_v2([x, z_id], 512)

    x = id_mod_res_v2([x, z_id], 512)

    x = id_mod_res_v2([x, z_id], 512)

    if deep:
        x = UpSampling2D(interpolation='bilinear')(x)
        x = Conv2D(filters=512, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Activation(tf.keras.activations.relu)(x)

    x = UpSampling2D(interpolation='bilinear')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)

    x = UpSampling2D(interpolation='bilinear')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)                         # 56

    x = UpSampling2D(interpolation='bilinear')(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Activation(tf.keras.activations.relu)(x)                       # 112

    x = ReflectionPadding2D(padding=(3, 3))(x)
    out = Conv2D(filters=3, kernel_size=7, padding='valid')(x)            # 112
    out = Activation('sigmoid')(out)

    model = Model([inputs, z_id], out)
    model.summary()

    return model


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


class FilteredReLU(Layer):

    def __init__(self,
                 critically_sampled,

                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,

                 conv_kernel        = 3,
                 lrelu_upsampling   = 2,
                 filter_size        = 6,
                 conv_clamp         = 256,
                 use_radial_filters = False,
                 is_torgb           = False,
                 **kwargs):
        super(FilteredReLU, self).__init__(**kwargs)
        self.critically_sampled = critically_sampled

        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.in_size            = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size           = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate   = in_sampling_rate
        self.out_sampling_rate  = out_sampling_rate
        self.in_cutoff          = in_cutoff
        self.out_cutoff         = out_cutoff
        self.in_half_width      = in_half_width
        self.out_half_width     = out_half_width

        self.is_torgb = is_torgb

        self.conv_kernel        = 1 if is_torgb else conv_kernel
        self.lrelu_upsampling   = lrelu_upsampling
        self.conv_clamp         = conv_clamp

        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)

        # Up sampling filter
        self.u_factor           = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.u_factor == self.tmp_sampling_rate
        self.u_taps             = filter_size * self.u_factor if self.u_factor > 1 and not self.is_torgb else 1
        self.u_filter           = self.design_lowpass_filter(numtaps=self.u_taps,
                                                             cutoff=self.in_cutoff,
                                                             width=self.in_half_width*2,
                                                             fs=self.tmp_sampling_rate)

        # Down sampling filter
        self.d_factor           = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.d_factor == self.tmp_sampling_rate
        self.d_taps             = filter_size * self.d_factor if self.d_factor > 1 and not self.is_torgb else 1
        self.d_radial           = use_radial_filters and not self.critically_sampled
        self.d_filter           = self.design_lowpass_filter(numtaps=self.d_taps,
                                                             cutoff=self.out_cutoff,
                                                             width=self.out_half_width*2,
                                                             fs=self.tmp_sampling_rate,
                                                             radial=self.d_radial)
        # Compute padding
        pad_total               = (self.out_size - 1) * self.d_factor + 1
        pad_total              -= (self.in_size + self.conv_kernel - 1) * self.u_factor
        pad_total              += self.u_taps + self.d_taps - 2
        pad_lo                  = (pad_total + self.u_factor) // 2
        pad_hi                  = pad_total - pad_lo
        self.padding            = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        self.gain               = 1 if self.is_torgb else np.sqrt(2)
        self.slope              = 1 if self.is_torgb else 0.2

        self.act_funcs          = {'linear':
                                       {'func': lambda x, **_: x,
                                        'def_alpha': 0,
                                        'def_gain': 1},
                                   'lrelu':
                                       {'func': lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha),
                                        'def_alpha': 0.2,
                                        'def_gain': np.sqrt(2)},
                                   }

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(out_channels,),
                                                     dtype="float32"),
                                trainable=True)

    def design_lowpass_filter(self, numtaps, cutoff, width, fs, radial=False):
        if numtaps == 1:
            return None

        if not radial:
            f = sps.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return f

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = spspec.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = sps.kaiser_beta(sps.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return f

    def get_filter_size(self, f):
        if f is None:
            return 1, 1
        assert 1 <= f.ndim <= 2
        return f.shape[-1], f.shape[0]  # width, height

    def parse_padding(self, padding):
        if isinstance(padding, int):
            padding = [padding, padding]
        assert isinstance(padding, (list, tuple))
        assert all(isinstance(x, (int, np.integer)) for x in padding)
        padding = [int(x) for x in padding]
        if len(padding) == 2:
            px, py = padding
            padding = [px, px, py, py]
        px0, px1, py0, py1 = padding
        return px0, px1, py0, py1

    def bias_act(self, x, b=None, dim=3, act='linear', alpha=None, gain=None, clamp=None):
        spec = self.act_funcs[act]
        alpha = float(alpha if alpha is not None else spec['def_alpha'])
        gain = float(gain if gain is not None else spec['def_gain'])
        clamp = float(clamp if clamp is not None else -1)

        if b is not None:
            x = x + tf.reshape(b, shape=[-1 if i == dim else 1 for i in range(len(x.shape))])
        x = spec['func'](x, alpha=alpha)

        if gain != 1:
            x = x * gain

        if clamp >= 0:
            x = tf.clip_by_value(x, -clamp, clamp)
        return x

    def parse_scaling(self, scaling):
        if isinstance(scaling, int):
            scaling = [scaling, scaling]
        sx, sy = scaling
        assert sx >= 1 and sy >= 1
        return sx, sy

    def upfirdn2d(self, x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
        if f is None:
            f = tf.ones([1, 1], dtype=tf.float32)

        batch_size, in_height, in_width, num_channels = x.shape

        upx, upy = self.parse_scaling(up)
        downx, downy = self.parse_scaling(down)
        padx0, padx1, pady0, pady1 = self.parse_padding(padding)

        upW = in_width * upx + padx0 + padx1
        upH = in_height * upy + pady0 + pady1
        assert upW >= f.shape[-1] and upH >= f.shape[0]

        # Channel first format.
        x = tf.transpose(x, perm=[0, 3, 1, 2])

        # Upsample by inserting zeros.
        x = tf.reshape(x, [num_channels, batch_size, in_height, 1, in_width, 1])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, upx - 1], [0, 0], [0, upy - 1]])
        x = tf.reshape(x, [batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = tf.pad(x, [[0, 0], [0, 0],
                       [tf.math.maximum(padx0, 0), tf.math.maximum(padx1, 0)],
                       [tf.math.maximum(pady0, 0), tf.math.maximum(pady1, 0)]])
        x = x[:, :,
            tf.math.maximum(-pady0, 0) : x.shape[2] - tf.math.maximum(-pady1, 0),
            tf.math.maximum(-padx0, 0) : x.shape[3] - tf.math.maximum(-padx1, 0)]

        # Setup filter.
        f = f * (gain ** (f.ndim / 2))
        f = tf.cast(f, dtype=x.dtype)
        if not flip_filter:
            f = tf.reverse(f, axis=[-1])
        f = tf.reshape(f, shape=(1, 1, f.shape[-1]))
        f = tf.repeat(f, repeats=num_channels, axis=0)

        if tf.rank(f) == 4:
            f_0 = tf.transpose(f, perm=[2, 3, 1, 0])
            x = tf.nn.conv2d(x, f_0, 1, 'VALID')
        else:
            f_0 = tf.expand_dims(f, axis=2)
            f_0 = tf.transpose(f_0, perm=[2, 3, 1, 0])

            f_1 = tf.expand_dims(f, axis=3)
            f_1 = tf.transpose(f_1, perm=[2, 3, 1, 0])

            x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
            x = tf.nn.conv2d(x, f_1, 1, 'VALID', data_format='NCHW')

        x = x[:, :, ::downy, ::downx]

        # Back to channel last.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x


    def filtered_lrelu(self,
                       x, fu=None, fd=None, b=None,
                       up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
        #fu_w, fu_h = self.get_filter_size(fu)
        #fd_w, fd_h = self.get_filter_size(fd)

        px0, px1, py0, py1 = self.parse_padding(padding)

        #batch_size, in_h, in_w, channels = x.shape
        #in_dtype = x.dtype
        #out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
        #out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

        x = self.bias_act(x=x, b=b)
        x = self.upfirdn2d(x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)
        x = self.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)
        x = self.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)

        return x


    def call(self, inputs):
        return self.filtered_lrelu(inputs,
                                   fu=self.u_filter,
                                   fd=self.d_filter,
                                   b=self.bias,
                                   up=self.u_factor,
                                   down=self.d_factor,
                                   padding=self.padding,
                                   gain=self.gain,
                                   slope=self.slope,
                                   clamp=self.conv_clamp)

    def get_config(self):
        base_config = super(FilteredReLU, self).get_config()
        return base_config


class SynthesisLayer(Layer):

    def __init__(self,
                 critically_sampled,

                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,

                 conv_kernel        = 3,
                 lrelu_upsampling   = 2,
                 filter_size        = 6,
                 conv_clamp         = 256,
                 use_radial_filters = False,
                 is_torgb           = False,
                 **kwargs):
        super(SynthesisLayer, self).__init__(**kwargs)
        self.critically_sampled = critically_sampled

        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.in_size            = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size           = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate   = in_sampling_rate
        self.out_sampling_rate  = out_sampling_rate
        self.in_cutoff          = in_cutoff
        self.out_cutoff         = out_cutoff
        self.in_half_width      = in_half_width
        self.out_half_width     = out_half_width

        self.is_torgb = is_torgb

        self.conv_kernel        = 1 if is_torgb else conv_kernel
        self.lrelu_upsampling   = lrelu_upsampling
        self.conv_clamp         = conv_clamp

        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)

        # Up sampling filter
        self.u_factor           = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.u_factor == self.tmp_sampling_rate
        self.u_taps             = filter_size * self.u_factor if self.u_factor > 1 and not self.is_torgb else 1
        self.u_filter           = self.design_lowpass_filter(numtaps=self.u_taps,
                                                             cutoff=self.in_cutoff,
                                                             width=self.in_half_width*2,
                                                             fs=self.tmp_sampling_rate)

        # Down sampling filter
        self.d_factor           = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.d_factor == self.tmp_sampling_rate
        self.d_taps             = filter_size * self.d_factor if self.d_factor > 1 and not self.is_torgb else 1
        self.d_radial           = use_radial_filters and not self.critically_sampled
        self.d_filter           = self.design_lowpass_filter(numtaps=self.d_taps,
                                                             cutoff=self.out_cutoff,
                                                             width=self.out_half_width*2,
                                                             fs=self.tmp_sampling_rate,
                                                             radial=self.d_radial)
        # Compute padding
        pad_total               = (self.out_size - 1) * self.d_factor + 1
        pad_total              -= (self.in_size + self.conv_kernel - 1) * self.u_factor
        pad_total              += self.u_taps + self.d_taps - 2
        pad_lo                  = (pad_total + self.u_factor) // 2
        pad_hi                  = pad_total - pad_lo
        self.padding            = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        self.gain               = 1 if self.is_torgb else np.sqrt(2)
        self.slope              = 1 if self.is_torgb else 0.2

        self.act_funcs          = {'linear':
                                       {'func': lambda x, **_: x,
                                        'def_alpha': 0,
                                        'def_gain': 1},
                                   'lrelu':
                                       {'func': lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha),
                                        'def_alpha': 0.2,
                                        'def_gain': np.sqrt(2)},
                                   }

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(out_channels,),
                                                     dtype="float32"),
                                trainable=True)
        self.affine = Dense(self.in_channels)
        self.conv = Conv2DMod(self.out_channels, kernel_size=self.conv_kernel, padding='same')

    def design_lowpass_filter(self, numtaps, cutoff, width, fs, radial=False):
        if numtaps == 1:
            return None

        if not radial:
            f = sps.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return f

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = spspec.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = sps.kaiser_beta(sps.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return f

    def get_filter_size(self, f):
        if f is None:
            return 1, 1
        assert 1 <= f.ndim <= 2
        return f.shape[-1], f.shape[0]  # width, height

    def parse_padding(self, padding):
        if isinstance(padding, int):
            padding = [padding, padding]
        assert isinstance(padding, (list, tuple))
        assert all(isinstance(x, (int, np.integer)) for x in padding)
        padding = [int(x) for x in padding]
        if len(padding) == 2:
            px, py = padding
            padding = [px, px, py, py]
        px0, px1, py0, py1 = padding
        return px0, px1, py0, py1

    def bias_act(self, x, b=None, dim=3, act='linear', alpha=None, gain=None, clamp=None):
        spec = self.act_funcs[act]
        alpha = float(alpha if alpha is not None else spec['def_alpha'])
        gain = float(gain if gain is not None else spec['def_gain'])
        clamp = float(clamp if clamp is not None else -1)

        if b is not None:
            x = x + tf.reshape(b, shape=[-1 if i == dim else 1 for i in range(len(x.shape))])
        x = spec['func'](x, alpha=alpha)

        if gain != 1:
            x = x * gain

        if clamp >= 0:
            x = tf.clip_by_value(x, -clamp, clamp)
        return x

    def parse_scaling(self, scaling):
        if isinstance(scaling, int):
            scaling = [scaling, scaling]
        sx, sy = scaling
        assert sx >= 1 and sy >= 1
        return sx, sy

    def upfirdn2d(self, x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
        if f is None:
            f = tf.ones([1, 1], dtype=tf.float32)

        batch_size, in_height, in_width, num_channels = x.shape

        upx, upy = self.parse_scaling(up)
        downx, downy = self.parse_scaling(down)
        padx0, padx1, pady0, pady1 = self.parse_padding(padding)

        upW = in_width * upx + padx0 + padx1
        upH = in_height * upy + pady0 + pady1
        assert upW >= f.shape[-1] and upH >= f.shape[0]

        # Channel first format.
        x = tf.transpose(x, perm=[0, 3, 1, 2])

        # Upsample by inserting zeros.
        x = tf.reshape(x, [num_channels, batch_size, in_height, 1, in_width, 1])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, upx - 1], [0, 0], [0, upy - 1]])
        x = tf.reshape(x, [batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = tf.pad(x, [[0, 0], [0, 0],
                       [tf.math.maximum(padx0, 0), tf.math.maximum(padx1, 0)],
                       [tf.math.maximum(pady0, 0), tf.math.maximum(pady1, 0)]])
        x = x[:, :, tf.math.maximum(-pady0, 0) : x.shape[2] - tf.math.maximum(-pady1, 0), tf.math.maximum(-padx0, 0) : x.shape[3] - tf.math.maximum(-padx1, 0)]

        # Setup filter.
        f = f * (gain ** (f.ndim / 2))
        f = tf.cast(f, dtype=x.dtype)
        if not flip_filter:
            f = tf.reverse(f, axis=[-1])
        f = tf.reshape(f, shape=(1, 1, f.shape[-1]))
        f = tf.repeat(f, repeats=num_channels, axis=0)

        if tf.rank(f) == 4:
            f_0 = tf.transpose(f, perm=[2, 3, 1, 0])
            x = tf.nn.conv2d(x, f_0, 1, 'VALID')
        else:
            f_0 = tf.expand_dims(f, axis=2)
            f_0 = tf.transpose(f_0, perm=[2, 3, 1, 0])

            f_1 = tf.expand_dims(f, axis=3)
            f_1 = tf.transpose(f_1, perm=[2, 3, 1, 0])

            x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
            x = tf.nn.conv2d(x, f_1, 1, 'VALID', data_format='NCHW')

        x = x[:, :, ::downy, ::downx]

        # Back to channel last.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x


    def filtered_lrelu(self,
                       x, fu=None, fd=None, b=None,
                       up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
        #fu_w, fu_h = self.get_filter_size(fu)
        #fd_w, fd_h = self.get_filter_size(fd)

        px0, px1, py0, py1 = self.parse_padding(padding)

        #batch_size, in_h, in_w, channels = x.shape
        #in_dtype = x.dtype
        #out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
        #out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

        x = self.bias_act(x=x, b=b)
        x = self.upfirdn2d(x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)
        x = self.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)
        x = self.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)

        return x


    def call(self, inputs):
        x, w = inputs
        styles = self.affine(w)
        x = self.conv([x, styles])
        x = self.filtered_lrelu(x,
                                   fu=self.u_filter,
                                   fd=self.d_filter,
                                   b=self.bias,
                                   up=self.u_factor,
                                   down=self.d_factor,
                                   padding=self.padding,
                                   gain=self.gain,
                                   slope=self.slope,
                                   clamp=self.conv_clamp)
        return x

    def get_config(self):
        base_config = super(SynthesisLayer, self).get_config()
        return base_config


class SynthesisLayerNoMod(Layer):

    def __init__(self,
                 critically_sampled,

                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,

                 conv_kernel        = 3,
                 lrelu_upsampling   = 2,
                 filter_size        = 6,
                 conv_clamp         = 256,
                 use_radial_filters = False,
                 is_torgb           = False,
                 batch_size         = 10,
                 **kwargs):
        super(SynthesisLayerNoMod, self).__init__(**kwargs)
        self.critically_sampled = critically_sampled
        self.bs                 = batch_size

        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.in_size            = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size           = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate   = in_sampling_rate
        self.out_sampling_rate  = out_sampling_rate
        self.in_cutoff          = in_cutoff
        self.out_cutoff         = out_cutoff
        self.in_half_width      = in_half_width
        self.out_half_width     = out_half_width

        self.is_torgb = is_torgb

        self.conv_kernel        = 1 if is_torgb else conv_kernel
        self.lrelu_upsampling   = lrelu_upsampling
        self.conv_clamp         = conv_clamp

        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)

        # Up sampling filter
        self.u_factor           = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.u_factor == self.tmp_sampling_rate
        self.u_taps             = filter_size * self.u_factor if self.u_factor > 1 and not self.is_torgb else 1
        self.u_filter           = self.design_lowpass_filter(numtaps=self.u_taps,
                                                             cutoff=self.in_cutoff,
                                                             width=self.in_half_width*2,
                                                             fs=self.tmp_sampling_rate)

        # Down sampling filter
        self.d_factor           = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.d_factor == self.tmp_sampling_rate
        self.d_taps             = filter_size * self.d_factor if self.d_factor > 1 and not self.is_torgb else 1
        self.d_radial           = use_radial_filters and not self.critically_sampled
        self.d_filter           = self.design_lowpass_filter(numtaps=self.d_taps,
                                                             cutoff=self.out_cutoff,
                                                             width=self.out_half_width*2,
                                                             fs=self.tmp_sampling_rate,
                                                             radial=self.d_radial)
        # Compute padding
        pad_total               = (self.out_size - 1) * self.d_factor + 1
        pad_total              -= (self.in_size + self.conv_kernel - 1) * self.u_factor
        pad_total              += self.u_taps + self.d_taps - 2
        pad_lo                  = (pad_total + self.u_factor) // 2
        pad_hi                  = pad_total - pad_lo
        self.padding            = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        self.gain               = 1 if self.is_torgb else np.sqrt(2)
        self.slope              = 1 if self.is_torgb else 0.2

        self.act_funcs          = {'linear':
                                       {'func': lambda x, **_: x,
                                        'def_alpha': 0,
                                        'def_gain': 1},
                                   'lrelu':
                                       {'func': lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha),
                                        'def_alpha': 0.2,
                                        'def_gain': np.sqrt(2)},
                                   }

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(out_channels,),
                                                     dtype="float32"),
                                trainable=True)
        self.conv = Conv2D(self.out_channels, kernel_size=self.conv_kernel, padding='same')

    def design_lowpass_filter(self, numtaps, cutoff, width, fs, radial=False):
        if numtaps == 1:
            return None

        if not radial:
            f = sps.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return f

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = spspec.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = sps.kaiser_beta(sps.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return f

    def get_filter_size(self, f):
        if f is None:
            return 1, 1
        assert 1 <= f.ndim <= 2
        return f.shape[-1], f.shape[0]  # width, height

    def parse_padding(self, padding):
        if isinstance(padding, int):
            padding = [padding, padding]
        assert isinstance(padding, (list, tuple))
        assert all(isinstance(x, (int, np.integer)) for x in padding)
        padding = [int(x) for x in padding]
        if len(padding) == 2:
            px, py = padding
            padding = [px, px, py, py]
        px0, px1, py0, py1 = padding
        return px0, px1, py0, py1

    @tf.function
    def bias_act(self, x, b=None, dim=3, act='linear', alpha=None, gain=None, clamp=None):
        spec = self.act_funcs[act]
        alpha = float(alpha if alpha is not None else spec['def_alpha'])
        gain = float(gain if gain is not None else spec['def_gain'])
        clamp = float(clamp if clamp is not None else -1)

        if b is not None:
            x = x + tf.reshape(b, shape=[-1 if i == dim else 1 for i in range(len(x.shape))])
        x = spec['func'](x, alpha=alpha)

        if gain != 1:
            x = x * gain

        if clamp >= 0:
            x = tf.clip_by_value(x, -clamp, clamp)
        return x

    def parse_scaling(self, scaling):
        if isinstance(scaling, int):
            scaling = [scaling, scaling]
        sx, sy = scaling
        assert sx >= 1 and sy >= 1
        return sx, sy

    @tf.function
    def upfirdn2d(self, x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
        if f is None:
            f = tf.ones([1, 1], dtype=tf.float32)

        batch_size, in_height, in_width, num_channels = x.shape
        batch_size = tf.shape(x)[0]

        upx, upy = self.parse_scaling(up)
        downx, downy = self.parse_scaling(down)
        padx0, padx1, pady0, pady1 = self.parse_padding(padding)

        upW = in_width * upx + padx0 + padx1
        upH = in_height * upy + pady0 + pady1
        assert upW >= f.shape[-1] and upH >= f.shape[0]

        # Channel first format.
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Upsample by inserting zeros.
        x = tf.reshape(x, [batch_size, num_channels, in_height, 1, in_width, 1])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, upx - 1], [0, 0], [0, upy - 1]])
        x = tf.reshape(x, [batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = tf.pad(x, [[0, 0], [0, 0],
                       [tf.math.maximum(padx0, 0), tf.math.maximum(padx1, 0)],
                       [tf.math.maximum(pady0, 0), tf.math.maximum(pady1, 0)]])
        x = x[:, :,
            tf.math.maximum(-pady0, 0) : x.shape[2] - tf.math.maximum(-pady1, 0),
            tf.math.maximum(-padx0, 0) : x.shape[3] - tf.math.maximum(-padx1, 0)]

        # Setup filter.
        f = f * (gain ** (tf.rank(f) / 2))
        f = tf.cast(f, dtype=x.dtype)
        if not flip_filter:
            f = tf.reverse(f, axis=[-1])
        f = tf.reshape(f, shape=(1, 1, f.shape[-1]))
        f = tf.repeat(f, repeats=num_channels, axis=0)

        #if tf.rank(f) == 500:
        #    f_0 = tf.transpose(f, perm=[2, 3, 1, 0])
        #    x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
        #else:
        f_0 = tf.expand_dims(f, axis=2)
        f_0 = tf.transpose(f_0, perm=[2, 3, 1, 0])

        f_1 = tf.expand_dims(f, axis=3)
        f_1 = tf.transpose(f_1, perm=[2, 3, 1, 0])

        x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
        x = tf.nn.conv2d(x, f_1, 1, 'VALID', data_format='NCHW')

        x = x[:, :, ::downy, ::downx]

        # Back to channel last.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x

    @tf.function
    def filtered_lrelu(self,
                       x, fu=None, fd=None, b=None,
                       up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
        #fu_w, fu_h = self.get_filter_size(fu)
        #fd_w, fd_h = self.get_filter_size(fd)

        px0, px1, py0, py1 = self.parse_padding(padding)

        #batch_size, in_h, in_w, channels = x.shape
        #in_dtype = x.dtype
        #out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
        #out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

        x = self.bias_act(x=x, b=b)
        x = self.upfirdn2d(x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)
        x = self.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)
        x = self.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)

        return x


    def call(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.filtered_lrelu(x,
                                   fu=self.u_filter,
                                   fd=self.d_filter,
                                   b=self.bias,
                                   up=self.u_factor,
                                   down=self.d_factor,
                                   padding=self.padding,
                                   gain=self.gain,
                                   slope=self.slope,
                                   clamp=self.conv_clamp)
        return x

    def get_config(self):
        base_config = super(SynthesisLayer, self).get_config()
        return base_config


class SynthesisLayerNoModBN(Layer):

    def __init__(self,
                 critically_sampled,

                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,

                 conv_kernel        = 3,
                 lrelu_upsampling   = 2,
                 filter_size        = 6,
                 conv_clamp         = 256,
                 use_radial_filters = False,
                 is_torgb           = False,
                 batch_size         = 10,
                 **kwargs):
        super(SynthesisLayerNoModBN, self).__init__(**kwargs)
        self.critically_sampled = critically_sampled
        self.bs                 = batch_size

        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.in_size            = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size           = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate   = in_sampling_rate
        self.out_sampling_rate  = out_sampling_rate
        self.in_cutoff          = in_cutoff
        self.out_cutoff         = out_cutoff
        self.in_half_width      = in_half_width
        self.out_half_width     = out_half_width

        self.is_torgb = is_torgb

        self.conv_kernel        = 1 if is_torgb else conv_kernel
        self.lrelu_upsampling   = lrelu_upsampling
        self.conv_clamp         = conv_clamp

        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1.0 if is_torgb else lrelu_upsampling)

        # Up sampling filter
        self.u_factor           = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.u_factor == self.tmp_sampling_rate
        self.u_taps             = filter_size * self.u_factor if self.u_factor > 1 and not self.is_torgb else 1
        self.u_filter           = self.design_lowpass_filter(numtaps=self.u_taps,
                                                             cutoff=self.in_cutoff,
                                                             width=self.in_half_width*2,
                                                             fs=self.tmp_sampling_rate)

        # Down sampling filter
        self.d_factor           = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.d_factor == self.tmp_sampling_rate
        self.d_taps             = filter_size * self.d_factor if self.d_factor > 1 and not self.is_torgb else 1
        self.d_radial           = use_radial_filters and not self.critically_sampled
        self.d_filter           = self.design_lowpass_filter(numtaps=self.d_taps,
                                                             cutoff=self.out_cutoff,
                                                             width=self.out_half_width*2,
                                                             fs=self.tmp_sampling_rate,
                                                             radial=self.d_radial)
        # Compute padding
        pad_total               = (self.out_size - 1) * self.d_factor + 1
        pad_total              -= (self.in_size) * self.u_factor
        pad_total              += self.u_taps + self.d_taps - 2
        pad_lo                  = (pad_total + self.u_factor) // 2
        pad_hi                  = pad_total - pad_lo
        self.padding            = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        self.gain               = 1 if self.is_torgb else np.sqrt(2)
        self.slope              = 1 if self.is_torgb else 0.2

        self.act_funcs          = {'linear':
                                       {'func': lambda x, **_: x,
                                        'def_alpha': 0,
                                        'def_gain': 1},
                                   'lrelu':
                                       {'func': lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha),
                                        'def_alpha': 0.2,
                                        'def_gain': np.sqrt(2)},
                                   }

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(out_channels,),
                                                     dtype="float32"),
                                trainable=True)
        self.conv = Conv2D(self.out_channels, kernel_size=self.conv_kernel, padding='same')
        self.bn = BatchNormalization()

    def design_lowpass_filter(self, numtaps, cutoff, width, fs, radial=False):
        if numtaps == 1:
            return None

        if not radial:
            f = sps.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return f

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = spspec.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = sps.kaiser_beta(sps.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return f

    def get_filter_size(self, f):
        if f is None:
            return 1, 1
        assert 1 <= f.ndim <= 2
        return f.shape[-1], f.shape[0]  # width, height

    def parse_padding(self, padding):
        if isinstance(padding, int):
            padding = [padding, padding]
        assert isinstance(padding, (list, tuple))
        assert all(isinstance(x, (int, np.integer)) for x in padding)
        padding = [int(x) for x in padding]
        if len(padding) == 2:
            px, py = padding
            padding = [px, px, py, py]
        px0, px1, py0, py1 = padding
        return px0, px1, py0, py1

    @tf.function
    def bias_act(self, x, b=None, dim=3, act='linear', alpha=None, gain=None, clamp=None):
        spec = self.act_funcs[act]
        alpha = float(alpha if alpha is not None else spec['def_alpha'])
        gain = tf.cast(gain if gain is not None else spec['def_gain'], tf.float32)
        clamp = float(clamp if clamp is not None else -1)

        if b is not None:
            x = x + tf.reshape(b, shape=[-1 if i == dim else 1 for i in range(len(x.shape))])
        x = spec['func'](x, alpha=alpha)

        x = x * gain

        if clamp >= 0:
            x = tf.clip_by_value(x, -clamp, clamp)
        return x

    def parse_scaling(self, scaling):
        if isinstance(scaling, int):
            scaling = [scaling, scaling]
        sx, sy = scaling
        assert sx >= 1 and sy >= 1
        return sx, sy

    @tf.function
    def upfirdn2d(self, x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
        if f is None:
            f = tf.ones([1, 1], dtype=tf.float32)

        batch_size, in_height, in_width, num_channels = x.shape
        batch_size = tf.shape(x)[0]

        upx, upy = self.parse_scaling(up)
        downx, downy = self.parse_scaling(down)
        padx0, padx1, pady0, pady1 = self.parse_padding(padding)

        upW = in_width * upx + padx0 + padx1
        upH = in_height * upy + pady0 + pady1
        assert upW >= f.shape[-1] and upH >= f.shape[0]

        # Channel first format.
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Upsample by inserting zeros.
        x = tf.reshape(x, [batch_size, num_channels, in_height, 1, in_width, 1])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, upx - 1], [0, 0], [0, upy - 1]])
        x = tf.reshape(x, [batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = tf.pad(x, [[0, 0], [0, 0],
                       [tf.math.maximum(padx0, 0), tf.math.maximum(padx1, 0)],
                       [tf.math.maximum(pady0, 0), tf.math.maximum(pady1, 0)]])
        x = x[:, :,
            tf.math.maximum(-pady0, 0) : x.shape[2] - tf.math.maximum(-pady1, 0),
            tf.math.maximum(-padx0, 0) : x.shape[3] - tf.math.maximum(-padx1, 0)]

        # Setup filter.
        f = f * (gain ** (tf.rank(f) / 2))
        f = tf.cast(f, dtype=x.dtype)
        if not flip_filter:
            f = tf.reverse(f, axis=[-1])
        f = tf.reshape(f, shape=(1, 1, f.shape[-1]))
        f = tf.repeat(f, repeats=num_channels, axis=0)

        #if tf.rank(f) == 500:
        #    f_0 = tf.transpose(f, perm=[2, 3, 1, 0])
        #    x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
        #else:
        f_0 = tf.expand_dims(f, axis=2)
        f_0 = tf.transpose(f_0, perm=[2, 3, 1, 0])

        f_1 = tf.expand_dims(f, axis=3)
        f_1 = tf.transpose(f_1, perm=[2, 3, 1, 0])

        x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
        x = tf.nn.conv2d(x, f_1, 1, 'VALID', data_format='NCHW')

        x = x[:, :, ::downy, ::downx]

        # Back to channel last.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x

    @tf.function
    def filtered_lrelu(self,
                       x, fu=None, fd=None, b=None,
                       up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
        #fu_w, fu_h = self.get_filter_size(fu)
        #fd_w, fd_h = self.get_filter_size(fd)

        px0, px1, py0, py1 = self.parse_padding(padding)

        #batch_size, in_h, in_w, channels = x.shape
        #in_dtype = x.dtype
        #out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
        #out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

        x = self.bias_act(x=x, b=b)
        x = self.upfirdn2d(x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)
        x = self.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)
        x = self.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)

        return x


    def call(self, inputs):
        x = inputs
        x = self.conv(x)
        x = self.bn(x)
        x = self.filtered_lrelu(x,
                                   fu=self.u_filter,
                                   fd=self.d_filter,
                                   b=self.bias,
                                   up=self.u_factor,
                                   down=self.d_factor,
                                   padding=self.padding,
                                   gain=self.gain,
                                   slope=self.slope,
                                   clamp=self.conv_clamp)
        return x

    def get_config(self):
        base_config = super(SynthesisLayerNoModBN, self).get_config()
        return base_config


class SynthesisInput(Layer):
    def __init__(self,
                 w_dim,
                 channels,
                 size,
                 sampling_rate,
                 bandwidth,
                 **kwargs):
        super(SynthesisInput, self).__init__(**kwargs)
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = np.random.normal(size=(int(channels[0]), 2))
        radii = np.sqrt(np.sum(np.square(freqs), axis=1, keepdims=True))
        freqs /= radii * np.power(np.exp(np.square(radii)), 0.25)
        freqs *= bandwidth
        phases = np.random.uniform(size=[int(channels[0])]) - 0.5

        # Setup parameters and buffers.
        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(shape=(self.channels, self.channels),
                                                     dtype="float32"), rainable=True)
        self.affine = Dense(4, kernel_initializer=tf.zeros_initializer, bias_initializer=tf.zeros_initializer)
        self.transform = tf.eye(3, 3)
        self.freqs = tf.constant(freqs)
        self.phases = tf.constant(phases)

    def call(self, w):
        # Batch dimension
        transforms = tf.expand_dims(self.transform, axis=0)
        freqs = tf.expand_dims(self.freqs, axis=0)
        phases = tf.expand_dims(self.phases, axis=0)

        # Apply learned transformation.
        t = self.affine(w) # t = (r_c, r_s, t_x, t_y)
        t = t / tf.linalg.norm(t[:, :2], axis=1, keepdims=True)
        # Inverse rotation wrt. resulting image.
        m_r = tf.repeat(tf.expand_dims(tf.eye(3), axis=0), repeats=w.shape[0], axis=0)
        m_r[:, 0, 0] = t[:, 0]  # r'_c
        m_r[:, 0, 1] = -t[:, 1]  # r'_s
        m_r[:, 1, 0] = t[:, 1]  # r'_s
        m_r[:, 1, 1] = t[:, 0]  # r'_c
        # Inverse translation wrt. resulting image.
        m_t = tf.repeat(tf.expand_dims(tf.eye(3), axis=0), repeats=w.shape[0], axis=0)
        m_t[:, 0, 2] = -t[:, 2]  # t'_x
        m_t[:, 1, 2] = -t[:, 3]  # t'_y
        transforms = m_r @ m_t @ transforms

        # Transform frequencies.
        phases = phases + tf.expand_dims(freqs @ transforms[:, :2, 2:], axis=2)
        freqs = freqs @ transforms[:, :2, :2]

        # Dampen out-of-band frequencies that may occur due to the user-specified transform.
        amplitudes = tf.clip_by_value(1 - (tf.linalg.norm(freqs, axis=1, keepdims=True) - self.bandwidth) / (self.sampling_rate / 2 - self.bandwidth), 0, 1)



    def get_config(self):
        base_config = super(SynthesisInput, self).get_config()
        return base_config


class SynthesisLayerFS(Layer):

    def __init__(self,
                 critically_sampled,

                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,

                 conv_kernel        = 3,
                 lrelu_upsampling   = 2,
                 filter_size        = 6,
                 conv_clamp         = 256,
                 use_radial_filters = False,
                 is_torgb           = False,
                 **kwargs):
        super(SynthesisLayerFS, self).__init__(**kwargs)
        self.critically_sampled = critically_sampled

        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.in_size            = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size           = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate   = in_sampling_rate
        self.out_sampling_rate  = out_sampling_rate
        self.in_cutoff          = in_cutoff
        self.out_cutoff         = out_cutoff
        self.in_half_width      = in_half_width
        self.out_half_width     = out_half_width

        self.is_torgb = is_torgb

        self.conv_kernel        = 1 if is_torgb else conv_kernel
        self.lrelu_upsampling   = lrelu_upsampling
        self.conv_clamp         = conv_clamp

        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)

        # Up sampling filter
        self.u_factor           = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.u_factor == self.tmp_sampling_rate
        self.u_taps             = filter_size * self.u_factor if self.u_factor > 1 and not self.is_torgb else 1
        self.u_filter           = self.design_lowpass_filter(numtaps=self.u_taps,
                                                             cutoff=self.in_cutoff,
                                                             width=self.in_half_width*2,
                                                             fs=self.tmp_sampling_rate)

        # Down sampling filter
        self.d_factor           = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.d_factor == self.tmp_sampling_rate
        self.d_taps             = filter_size * self.d_factor if self.d_factor > 1 and not self.is_torgb else 1
        self.d_radial           = use_radial_filters and not self.critically_sampled
        self.d_filter           = self.design_lowpass_filter(numtaps=self.d_taps,
                                                             cutoff=self.out_cutoff,
                                                             width=self.out_half_width*2,
                                                             fs=self.tmp_sampling_rate,
                                                             radial=self.d_radial)
        # Compute padding
        pad_total               = (self.out_size - 1) * self.d_factor + 1
        pad_total              -= (self.in_size + self.conv_kernel - 1) * self.u_factor
        pad_total              += self.u_taps + self.d_taps - 2
        pad_lo                  = (pad_total + self.u_factor) // 2
        pad_hi                  = pad_total - pad_lo
        self.padding            = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        self.gain               = 1 if self.is_torgb else np.sqrt(2)
        self.slope              = 1 if self.is_torgb else 0.2

        self.act_funcs          = {'linear':
                                       {'func': lambda x, **_: x,
                                        'def_alpha': 0,
                                        'def_gain': 1},
                                   'lrelu':
                                       {'func': lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha),
                                        'def_alpha': 0.2,
                                        'def_gain': np.sqrt(2)},
                                   }

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(shape=(out_channels,),
                                                     dtype="float32"),
                                trainable=True)
        self.affine = Dense(self.out_channels)
        self.conv_mod = Conv2DMod(self.out_channels, kernel_size=self.conv_kernel, padding='same')
        self.bn = BatchNormalization()
        self.conv_gamma = Conv2D(self.out_channels, kernel_size=1)
        self.conv_beta = Conv2D(self.out_channels, kernel_size=1)
        self.conv_gate = Conv2D(self.out_channels, kernel_size=1)
        self.conv_final = Conv2D(self.out_channels, kernel_size=self.conv_kernel, padding='same')


    def design_lowpass_filter(self, numtaps, cutoff, width, fs, radial=False):
        if numtaps == 1:
            return None

        if not radial:
            f = sps.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return f

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = spspec.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = sps.kaiser_beta(sps.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return f

    def get_filter_size(self, f):
        if f is None:
            return 1, 1
        assert 1 <= f.ndim <= 2
        return f.shape[-1], f.shape[0]  # width, height

    def parse_padding(self, padding):
        if isinstance(padding, int):
            padding = [padding, padding]
        assert isinstance(padding, (list, tuple))
        assert all(isinstance(x, (int, np.integer)) for x in padding)
        padding = [int(x) for x in padding]
        if len(padding) == 2:
            px, py = padding
            padding = [px, px, py, py]
        px0, px1, py0, py1 = padding
        return px0, px1, py0, py1

    @tf.function
    def bias_act(self, x, b=None, dim=3, act='linear', alpha=None, gain=None, clamp=None):
        spec = self.act_funcs[act]
        alpha = float(alpha if alpha is not None else spec['def_alpha'])
        gain = tf.cast(gain if gain is not None else spec['def_gain'], tf.float32)
        clamp = float(clamp if clamp is not None else -1)

        if b is not None:
            x = x + tf.reshape(b, shape=[-1 if i == dim else 1 for i in range(len(x.shape))])
        x = spec['func'](x, alpha=alpha)

        x = x * gain

        if clamp >= 0:
            x = tf.clip_by_value(x, -clamp, clamp)
        return x

    def parse_scaling(self, scaling):
        if isinstance(scaling, int):
            scaling = [scaling, scaling]
        sx, sy = scaling
        assert sx >= 1 and sy >= 1
        return sx, sy

    @tf.function
    def upfirdn2d(self, x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
        if f is None:
            f = tf.ones([1, 1], dtype=tf.float32)

        batch_size, in_height, in_width, num_channels = x.shape
        batch_size = tf.shape(x)[0]

        upx, upy = self.parse_scaling(up)
        downx, downy = self.parse_scaling(down)
        padx0, padx1, pady0, pady1 = self.parse_padding(padding)

        upW = in_width * upx + padx0 + padx1
        upH = in_height * upy + pady0 + pady1
        assert upW >= f.shape[-1] and upH >= f.shape[0]

        # Channel first format.
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        # Upsample by inserting zeros.
        x = tf.reshape(x, [batch_size, num_channels, in_height, 1, in_width, 1])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, upx - 1], [0, 0], [0, upy - 1]])
        x = tf.reshape(x, [batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = tf.pad(x, [[0, 0], [0, 0],
                       [tf.math.maximum(padx0, 0), tf.math.maximum(padx1, 0)],
                       [tf.math.maximum(pady0, 0), tf.math.maximum(pady1, 0)]])
        x = x[:, :,
            tf.math.maximum(-pady0, 0): x.shape[2] - tf.math.maximum(-pady1, 0),
            tf.math.maximum(-padx0, 0): x.shape[3] - tf.math.maximum(-padx1, 0)]

        # Setup filter.
        f = f * (gain ** (tf.rank(f) / 2))
        f = tf.cast(f, dtype=x.dtype)
        if not flip_filter:
            f = tf.reverse(f, axis=[-1])
        f = tf.reshape(f, shape=(1, 1, f.shape[-1]))
        f = tf.repeat(f, repeats=num_channels, axis=0)

        # if tf.rank(f) == 500:
        #    f_0 = tf.transpose(f, perm=[2, 3, 1, 0])
        #    x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
        # else:
        f_0 = tf.expand_dims(f, axis=2)
        f_0 = tf.transpose(f_0, perm=[2, 3, 1, 0])

        f_1 = tf.expand_dims(f, axis=3)
        f_1 = tf.transpose(f_1, perm=[2, 3, 1, 0])

        x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
        x = tf.nn.conv2d(x, f_1, 1, 'VALID', data_format='NCHW')

        x = x[:, :, ::downy, ::downx]

        # Back to channel last.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x

    @tf.function
    def filtered_lrelu(self,
                       x, fu=None, fd=None, b=None,
                       up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
        #fu_w, fu_h = self.get_filter_size(fu)
        #fd_w, fd_h = self.get_filter_size(fd)

        px0, px1, py0, py1 = self.parse_padding(padding)

        #batch_size, in_h, in_w, channels = x.shape
        #in_dtype = x.dtype
        #out_w = (in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)) // down
        #out_h = (in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)) // down

        x = self.bias_act(x=x, b=b)
        x = self.upfirdn2d(x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)
        x = self.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)
        x = self.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)

        return x

    @tf.function
    def aadm(self, x, w, a):
        w_affine = self.affine(w)
        x_norm = self.bn(x)

        x_id = self.conv_mod([x_norm, w_affine])

        gate = self.conv_gate(x_norm)
        gate = tf.nn.sigmoid(gate)

        x_att_beta = self.conv_beta(a)
        x_att_gamma = self.conv_gamma(a)

        x_att = x_norm * x_att_beta + x_att_gamma

        h = x_id * gate + (1 - gate) * x_att

        return h


    def call(self, inputs):
        x, w, a = inputs
        x = self.conv_final(x)
        x = self.aadm(x, w, a)
        x = self.filtered_lrelu(x,
                                   fu=self.u_filter,
                                   fd=self.d_filter,
                                   b=self.bias,
                                   up=self.u_factor,
                                   down=self.d_factor,
                                   padding=self.padding,
                                   gain=self.gain,
                                   slope=self.slope,
                                   clamp=self.conv_clamp)
        return x

    def get_config(self):
        base_config = super(SynthesisLayerFS, self).get_config()
        return base_config


class SynthesisLayerUpDownOnly(Layer):

    def __init__(self,
                 critically_sampled,

                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 in_sampling_rate,
                 out_sampling_rate,
                 in_cutoff,
                 out_cutoff,
                 in_half_width,
                 out_half_width,

                 conv_kernel        = 3,
                 lrelu_upsampling   = 2,
                 filter_size        = 6,
                 conv_clamp         = 256,
                 use_radial_filters = False,
                 is_torgb           = False,
                 **kwargs):
        super(SynthesisLayerUpDownOnly, self).__init__(**kwargs)
        self.critically_sampled = critically_sampled

        self.in_channels        = in_channels
        self.out_channels       = out_channels
        self.in_size            = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size           = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate   = in_sampling_rate
        self.out_sampling_rate  = out_sampling_rate
        self.in_cutoff          = in_cutoff
        self.out_cutoff         = out_cutoff
        self.in_half_width      = in_half_width
        self.out_half_width     = out_half_width

        self.is_torgb = is_torgb

        self.conv_kernel        = 1 if is_torgb else conv_kernel
        self.lrelu_upsampling   = lrelu_upsampling
        self.conv_clamp         = conv_clamp

        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)

        # Up sampling filter
        self.u_factor           = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.u_factor == self.tmp_sampling_rate
        self.u_taps             = filter_size * self.u_factor if self.u_factor > 1 and not self.is_torgb else 1
        self.u_filter           = self.design_lowpass_filter(numtaps=self.u_taps,
                                                             cutoff=self.in_cutoff,
                                                             width=self.in_half_width*2,
                                                             fs=self.tmp_sampling_rate)

        # Down sampling filter
        self.d_factor           = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.d_factor == self.tmp_sampling_rate
        self.d_taps             = filter_size * self.d_factor if self.d_factor > 1 and not self.is_torgb else 1
        self.d_radial           = use_radial_filters and not self.critically_sampled
        self.d_filter           = self.design_lowpass_filter(numtaps=self.d_taps,
                                                             cutoff=self.out_cutoff,
                                                             width=self.out_half_width*2,
                                                             fs=self.tmp_sampling_rate,
                                                             radial=self.d_radial)
        # Compute padding
        pad_total               = (self.out_size - 1) * self.d_factor + 1
        pad_total              -= (self.in_size + self.conv_kernel - 1) * self.u_factor
        pad_total              += self.u_taps + self.d_taps - 2
        pad_lo                  = (pad_total + self.u_factor) // 2
        pad_hi                  = pad_total - pad_lo
        self.padding            = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

        self.gain               = 1 if self.is_torgb else np.sqrt(2)
        self.slope              = 1 if self.is_torgb else 0.2

        self.act_funcs          = {'linear':
                                       {'func': lambda x, **_: x,
                                        'def_alpha': 0,
                                        'def_gain': 1},
                                   'lrelu':
                                       {'func': lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha),
                                        'def_alpha': 0.2,
                                        'def_gain': np.sqrt(2)},
                                   }

    def design_lowpass_filter(self, numtaps, cutoff, width, fs, radial=False):
        if numtaps == 1:
            return None

        if not radial:
            f = sps.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return f

        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = spspec.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = sps.kaiser_beta(sps.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return f

    def get_filter_size(self, f):
        if f is None:
            return 1, 1
        assert 1 <= f.ndim <= 2
        return f.shape[-1], f.shape[0]  # width, height

    def parse_padding(self, padding):
        if isinstance(padding, int):
            padding = [padding, padding]
        assert isinstance(padding, (list, tuple))
        assert all(isinstance(x, (int, np.integer)) for x in padding)
        padding = [int(x) for x in padding]
        if len(padding) == 2:
            px, py = padding
            padding = [px, px, py, py]
        px0, px1, py0, py1 = padding
        return px0, px1, py0, py1

    def bias_act(self, x, b=None, dim=3, act='linear', alpha=None, gain=None, clamp=None):
        spec = self.act_funcs[act]
        alpha = float(alpha if alpha is not None else spec['def_alpha'])
        gain = float(gain if gain is not None else spec['def_gain'])
        clamp = float(clamp if clamp is not None else -1)

        if b is not None:
            x = x + tf.reshape(b, shape=[-1 if i == dim else 1 for i in range(len(x.shape))])
        x = spec['func'](x, alpha=alpha)

        if gain != 1:
            x = x * gain

        if clamp >= 0:
            x = tf.clip_by_value(x, -clamp, clamp)
        return x

    def parse_scaling(self, scaling):
        if isinstance(scaling, int):
            scaling = [scaling, scaling]
        sx, sy = scaling
        assert sx >= 1 and sy >= 1
        return sx, sy

    def upfirdn2d(self, x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
        if f is None:
            f = tf.ones([1, 1], dtype=tf.float32)

        batch_size, in_height, in_width, num_channels = x.shape

        upx, upy = self.parse_scaling(up)
        downx, downy = self.parse_scaling(down)
        padx0, padx1, pady0, pady1 = self.parse_padding(padding)

        upW = in_width * upx + padx0 + padx1
        upH = in_height * upy + pady0 + pady1
        assert upW >= f.shape[-1] and upH >= f.shape[0]

        # Channel first format.
        x = tf.transpose(x, perm=[0, 3, 1, 2])

        # Upsample by inserting zeros.
        x = tf.reshape(x, [num_channels, batch_size, in_height, 1, in_width, 1])
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, upx - 1], [0, 0], [0, upy - 1]])
        x = tf.reshape(x, [batch_size, num_channels, in_height * upy, in_width * upx])

        # Pad or crop.
        x = tf.pad(x, [[0, 0], [0, 0],
                       [tf.math.maximum(padx0, 0), tf.math.maximum(padx1, 0)],
                       [tf.math.maximum(pady0, 0), tf.math.maximum(pady1, 0)]])
        x = x[:, :, tf.math.maximum(-pady0, 0) : x.shape[2] - tf.math.maximum(-pady1, 0), tf.math.maximum(-padx0, 0) : x.shape[3] - tf.math.maximum(-padx1, 0)]

        # Setup filter.
        f = f * (gain ** (f.ndim / 2))
        f = tf.cast(f, dtype=x.dtype)
        if not flip_filter:
            f = tf.reverse(f, axis=[-1])
        f = tf.reshape(f, shape=(1, 1, f.shape[-1]))
        f = tf.repeat(f, repeats=num_channels, axis=0)

        if tf.rank(f) == 4:
            f_0 = tf.transpose(f, perm=[2, 3, 1, 0])
            x = tf.nn.conv2d(x, f_0, 1, 'VALID')
        else:
            f_0 = tf.expand_dims(f, axis=2)
            f_0 = tf.transpose(f_0, perm=[2, 3, 1, 0])

            f_1 = tf.expand_dims(f, axis=3)
            f_1 = tf.transpose(f_1, perm=[2, 3, 1, 0])

            x = tf.nn.conv2d(x, f_0, 1, 'VALID', data_format='NCHW')
            x = tf.nn.conv2d(x, f_1, 1, 'VALID', data_format='NCHW')

        x = x[:, :, ::downy, ::downx]

        # Back to channel last.
        x = tf.transpose(x, perm=[0, 2, 3, 1])
        return x


    def filtered_lrelu(self,
                       x, fu=None, fd=None, b=None,
                       up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):

        px0, px1, py0, py1 = self.parse_padding(padding)

        x = self.upfirdn2d(x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up**2, flip_filter=flip_filter)
        x = self.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)
        x = self.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)

        return x


    def call(self, inputs):
        x = inputs
        x = self.filtered_lrelu(x,
                                   fu=self.u_filter,
                                   fd=self.d_filter,
                                   b=self.bias,
                                   up=self.u_factor,
                                   down=self.d_factor,
                                   padding=self.padding,
                                   gain=self.gain,
                                   slope=self.slope,
                                   clamp=self.conv_clamp)
        return x

    def get_config(self):
        base_config = super(SynthesisLayerUpDownOnly, self).get_config()
        return base_config


class Localization(Layer):
    def __init__(self):
        super(Localization, self).__init__()

        self.pool = MaxPooling2D()
        self.conv_0 = Conv2D(36, 5, activation='relu')
        self.conv_1 = Conv2D(36, 5, activation='relu')
        self.flatten = Flatten()
        self.fc_0 = Dense(36, activation='relu')
        self.fc_1 = Dense(6, bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                          kernel_initializer='zeros')
        self.reshape = Reshape((2, 3))

    def build(self, input_shape):
        print(input_shape)

    def compute_output_shape(self, input_shape):
        return [None, 6]

    def call(self, inputs):
        x = self.conv_0(inputs)
        x = self.pool(x)
        x = self.conv_1(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc_0(x)
        theta = self.fc_1(x)
        theta = self.reshape(theta)

        return theta


class BilinearInterpolation(Layer):
    def __init__(self, height=36, width=36):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        config = {
            'height': self.height,
            'width': self.width
        }
        base_config = super(BilinearInterpolation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def advance_indexing(self, inputs, x, y):
        shape = tf.shape(inputs)
        batch_size = shape[0]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))

        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(inputs, indices)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)

        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))

        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates

    def interpolate(self, images, homogenous_coordinates, theta):

        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])

            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]

            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) * tf.cast(self.height, dtype=tf.float32)) * 0.5

        with tf.name_scope("VaribleCasting"):
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, self.width-1)
            x1 = tf.clip_by_value(x1, 0, self.width - 1)
            y0 = tf.clip_by_value(y0, 0, self.height - 1)
            y1 = tf.clip_by_value(y1, 0, self.height - 1)
            x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32) - 1.0)
            y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32) - 1.0)

        with tf.name_scope("AdvancedIndexing"):
            i_a = self.advance_indexing(images, x0, y0)
            i_b = self.advance_indexing(images, x0, y1)
            i_c = self.advance_indexing(images, x1, y0)
            i_d = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"):
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)

            w_a = (x1 - x) * (y1 - y)
            w_b = (x1 - x) * (y - y0)
            w_c = (x - x0) * (y1 - y)
            w_d = (x - x0) * (y - y0)

            w_a = tf.expand_dims(w_a, axis=3)
            w_b = tf.expand_dims(w_b, axis=3)
            w_c = tf.expand_dims(w_c, axis=3)
            w_d = tf.expand_dims(w_d, axis=3)

        return tf.math.add_n([w_a * i_a + w_b * i_b + w_c * i_c + w_d * i_d])

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)


class ResBlockLR(Layer):
    def __init__(self, filters=16):
        super(ResBlockLR, self).__init__()
        self.filters = filters

        self.conv_0 = Conv2D(filters=filters,
                             kernel_size=3,
                             strides=1,
                             padding='same')
        self.bn_0 = BatchNormalization()
        self.conv_1 = Conv2D(filters=filters,
                             kernel_size=3,
                             strides=1,
                             padding='same')
        self.bn_1 = BatchNormalization()

    def get_config(self):
        config = {
            'filters': self.filters,
        }
        base_config = super(ResBlockLR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x = self.conv_0(inputs)
        x = self.bn_0(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.conv_1(x)
        x = self.bn_1(x)
        return x + inputs


class LearnedResize(Layer):
    def __init__(self, width, height, filters=16, in_channels=3, num_res_block=3, interpolation='bilinear'):
        super(LearnedResize, self).__init__()
        self.filters = filters
        self.num_res_block = num_res_block
        self.interpolation = interpolation
        self.in_channels = in_channels
        self.width = width
        self.height = height

        self.resize_layer = tf.keras.layers.experimental.preprocessing.Resizing(height,
                                                                                width,
                                                                                interpolation=interpolation)

        self.init_layers = tf.keras.models.Sequential([Conv2D(filters=filters,
                                                              kernel_size=7,
                                                              strides=1,
                                                              padding='same'),
                                                       LeakyReLU(0.2),
                                                       Conv2D(filters=filters,
                                                              kernel_size=1,
                                                              strides=1,
                                                              padding='same'),
                                                       LeakyReLU(0.2),
                                                       BatchNormalization()
                                                       ])
        res_blocks = []
        for i in range(num_res_block):
            res_blocks.append(ResBlockLR(filters=filters))
        res_blocks.append(Conv2D(filters=filters,
                                 kernel_size=3,
                                 strides=1,
                                 padding='same',
                                 use_bias=False))
        res_blocks.append(BatchNormalization())
        self.res_block_pipe = tf.keras.models.Sequential(res_blocks)
        self.final_conv = Conv2D(filters=in_channels,
                                 kernel_size=3,
                                 strides=1,
                                 padding='same')


    def compute_output_shape(self, input_shape):
        return [None, self.target_size[0], self.target_size[1], input_shape[-1]]

    def get_config(self):
        config = {
            'filters': self.filters,
            'num_res_block': self.num_res_block,
            'interpolation': self.interpolation,
            'width': self.width,
            'height': self.height,
        }
        base_config = super(LearnedResize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x_l = self.init_layers(inputs)
        x_l_0 = self.resize_layer(x_l)
        x_l = self.res_block_pipe(x_l_0)
        x_l = x_l + x_l_0
        x_l = self.final_conv(x_l)

        x = self.resize_layer(inputs)

        return x + x_l
