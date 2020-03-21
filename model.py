from functools import reduce

# from keras import layers
# from keras import initializers
# from keras import models
# from keras_ import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from keras_ import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import ClipBoxes, RegressBoxes, FilterDetections, wBiFPNAdd, BatchNormalization
from initializers import PriorProbability
from utils.anchors import anchors_for_shape
import numpy as np

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
d_bifpns = [3, 4, 5, 6, 7, 7, 8]
d_heads = [3, 3, 3, 4, 4, 4, 5]
image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]


def SeparableConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.SeparableConv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                                use_bias=True, name=f'{name}/conv')
    f2 = layers.BatchNormalization(momentum=0.997, epsilon=1e-4, name=f'{name}/bn')
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2))


def ConvBlock(num_channels, kernel_size, strides, name, freeze_bn=False):
    f1 = layers.Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same',
                       use_bias=True, name='{}_conv'.format(name))
    # f2 = BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f2 = layers.BatchNormalization(freeze=freeze_bn, name='{}_bn'.format(name))
    f3 = layers.ReLU(name='{}_relu'.format(name))
    return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), (f1, f2, f3))


def build_BiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5

        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same')(C5)
        P6_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4)(P6_in)
        P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_in)

        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_in)

        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P3_in)
        P3_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4)(P3_in)
        P4_in = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P4_in)
        P4_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4)(P4_in)
        P5_in = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P5_in)
        P5_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4)(P5_in)
        P6_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4)(P6_in)
        P7_in = layers.Conv2D(num_channels, kernel_size=1, padding='same')(P7_in)
        P7_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4)(P7_in)
    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features

    # upsample
    P7_U = layers.UpSampling2D()(P7_in)
    P6_td = layers.Add()([P7_U, P6_in])
    P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
    P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                               name='BiFPN_{}_U_P6'.format(id))(P6_td)
    P6_U = layers.UpSampling2D()(P6_td)
    P5_td = layers.Add()([P6_U, P5_in])
    P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
    P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                               name='BiFPN_{}_U_P5'.format(id))(P5_td)
    P5_U = layers.UpSampling2D()(P5_td)
    P4_td = layers.Add()([P5_U, P4_in])
    P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
    P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                               name='BiFPN_{}_U_P4'.format(id))(P4_td)
    P4_U = layers.UpSampling2D()(P4_td)
    P3_out = layers.Add()([P4_U, P3_in])
    P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
    P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                name='BiFPN_{}_U_P3'.format(id))(P3_out)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3_out)
    P4_out = layers.Add()([P3_D, P4_td, P4_in])
    P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
    P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                name='BiFPN_{}_D_P4'.format(id))(P4_out)
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4_out)
    P5_out = layers.Add()([P4_D, P5_td, P5_in])
    P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
    P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                name='BiFPN_{}_D_P5'.format(id))(P5_out)
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5_out)
    P6_out = layers.Add()([P5_D, P6_td, P6_in])
    P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
    P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                name='BiFPN_{}_D_P6'.format(id))(P6_out)
    P6_D = layers.MaxPooling2D(strides=(2, 2))(P6_out)
    P7_out = layers.Add()([P6_D, P7_in])
    P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
    P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1, freeze_bn=freeze_bn,
                                name='BiFPN_{}_D_P7'.format(id))(P7_out)

    return P3_out, P4_out, P5_out, P6_out, P7_out


def build_wBiFPN(features, num_channels, id, freeze_bn=False):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3_in = C3
        P4_in = C4
        P5_in = C5
        P6_in = layers.Conv2D(num_channels, kernel_size=1, padding='same', name='resample_p6/conv2d')(C5)
        P6_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4, name='resample_p6/bn')(P6_in)
        P6_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='resample_p6/maxpool')(P6_in)

        P7_in = layers.MaxPooling2D(pool_size=3, strides=2, padding='same',  name='resample_p7/maxpool')(P6_in)
        ####################################
        # upsample
        ####################################
        # fnode0
        P7_U = layers.UpSampling2D()(P7_in)
        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0')([P6_in, P7_U])
        P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        # fnode1
        P5_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/conv2d')(P5_in)
        P5_in_1 = layers.BatchNormalization(momentum=0.997, epsilon=1e-4,
                                            name=f'fpn_cells/cell_{id}/fnode1/resample_0_2_6/bn')(P5_in_1)
        P6_U = layers.UpSampling2D()(P6_td)
        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1')([P5_in_1, P6_U])
        P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        # fnode2
        P4_in_1 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/conv2d')(P4_in)
        P4_in_1 = layers.BatchNormalization(momentum=0.997, epsilon=1e-4,
                                            name=f'fpn_cells/cell_{id}/fnode2/resample_0_1_7/bn')(P4_in_1)
        P5_U = layers.UpSampling2D()(P5_td)
        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2')([P4_in_1, P5_U])
        P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        # fnode3
        P3_in = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                              name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/conv2d')(P3_in)
        P3_in = layers.BatchNormalization(momentum=0.997, epsilon=1e-4,
                                          name=f'fpn_cells/cell_{id}/fnode3/resample_0_0_8/bn')(P3_in)
        P4_U = layers.UpSampling2D()(P4_td)
        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3')([P3_in, P4_U])
        P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        ######################################
        # downsample
        ######################################
        # fnode4
        P4_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/conv2d')(P4_in)
        P4_in_2 = layers.BatchNormalization(momentum=0.997, epsilon=1e-4,
                                            name=f'fpn_cells/cell_{id}/fnode4/resample_0_1_9/bn')(P4_in_2)
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4')([P4_in_2, P4_td, P3_D])
        P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        # fnode5
        P5_in_2 = layers.Conv2D(num_channels, kernel_size=1, padding='same',
                                name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/conv2d')(P5_in)
        P5_in_2 = layers.BatchNormalization(momentum=0.997, epsilon=1e-4,
                                            name=f'fpn_cells/cell_{id}/fnode5/resample_0_2_10/bn')(P5_in_2)
        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5')([P5_in_2, P5_td, P4_D])
        P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        # fnode6
        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        # fnode7
        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7')([P7_in, P6_D])
        P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)

    else:
        P3_in, P4_in, P5_in, P6_in, P7_in = features
        ####################################
        # upsample
        ####################################
        # fnode0
        P7_U = layers.UpSampling2D()(P7_in)
        P6_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode0')([P6_in, P7_U])
        P6_td = layers.Activation(lambda x: tf.nn.swish(x))(P6_td)
        P6_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode0/op_after_combine5')(P6_td)
        # fnode1
        P6_U = layers.UpSampling2D()(P6_td)
        P5_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode1')([P5_in, P6_U])
        P5_td = layers.Activation(lambda x: tf.nn.swish(x))(P5_td)
        P5_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode1/op_after_combine6')(P5_td)
        # fnode2
        P5_U = layers.UpSampling2D()(P5_td)
        P4_td = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode2')([P4_in, P5_U])
        P4_td = layers.Activation(lambda x: tf.nn.swish(x))(P4_td)
        P4_td = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                   name=f'fpn_cells/cell_{id}/fnode2/op_after_combine7')(P4_td)
        # fnode3
        P4_U = layers.UpSampling2D()(P4_td)
        P3_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode3')([P3_in, P4_U])
        P3_out = layers.Activation(lambda x: tf.nn.swish(x))(P3_out)
        P3_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode3/op_after_combine8')(P3_out)
        ######################################
        # downsample
        ######################################
        # fnode4
        P3_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P3_out)
        P4_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode4')([P4_in, P4_td, P3_D])
        P4_out = layers.Activation(lambda x: tf.nn.swish(x))(P4_out)
        P4_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode4/op_after_combine9')(P4_out)

        # fnode5
        P4_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P4_out)
        P5_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode5')([P5_in, P5_td, P4_D])
        P5_out = layers.Activation(lambda x: tf.nn.swish(x))(P5_out)
        P5_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode5/op_after_combine10')(P5_out)

        # fnode6
        P5_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P5_out)
        P6_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode6')([P6_in, P6_td, P5_D])
        P6_out = layers.Activation(lambda x: tf.nn.swish(x))(P6_out)
        P6_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode6/op_after_combine11')(P6_out)

        # fnode7
        P6_D = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(P6_out)
        P7_out = wBiFPNAdd(name=f'fpn_cells/cell_{id}/fnode7')([P7_in, P6_D])
        P7_out = layers.Activation(lambda x: tf.nn.swish(x))(P7_out)
        P7_out = SeparableConvBlock(num_channels=num_channels, kernel_size=3, strides=1,
                                    name=f'fpn_cells/cell_{id}/fnode7/op_after_combine12')(P7_out)
    return P3_out, P4_td, P5_td, P6_td, P7_out


def build_regress_conv(width, depth, separable_conv=True):
    if separable_conv:
        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
    else:
        kernel_initializer = {
            'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'bias_initializer': 'zeros',
    }
    options.update(kernel_initializer)

    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        if separable_conv:
            outputs = layers.SeparableConv2D(filters=width, name=f'box-{i}', **options)(outputs)
        else:
            outputs = layers.Conv2D(filters=width, name=f'box-{i}', **options)(outputs)
    return models.Model(inputs=inputs, outputs=outputs, name='box_net_conv')


def build_regress_head(width, num_anchors=9, separable_conv=True, detect_quadrangle=False):
    if separable_conv:
        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
    else:
        kernel_initializer = {
            'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'bias_initializer': 'zeros',
    }
    options.update(kernel_initializer)
    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    if detect_quadrangle:
        if separable_conv:
            outputs = layers.SeparableConv2D(num_anchors * 9, name=f'box-predict', **options)(outputs)
        else:
            outputs = layers.Conv2D(num_anchors * 9, **options)(outputs)
        # (b, num_anchors_this_feature_map, 9)
        outputs = layers.Reshape((-1, 9))(outputs)
    else:
        if separable_conv:
            outputs = layers.SeparableConv2D(num_anchors * 9, name=f'box-predict', **options)(outputs)
        else:
            outputs = layers.Conv2D(num_anchors * 4, **options)(outputs)
        # (b, num_anchors_this_feature_map, 4)
        outputs = layers.Reshape((-1, 4))(outputs)

    return models.Model(inputs=inputs, outputs=outputs, name='box_net')


def build_regress_head(inputs, width, depth, level, num_anchors=9, detect_quadrangle=False, separable_conv=True):
    if separable_conv:
        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
    else:
        kernel_initializer = {
            'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'bias_initializer': 'zeros',
    }
    options.update(kernel_initializer)

    outputs = inputs
    for i in range(depth):
        if separable_conv:
            outputs = layers.SeparableConv2D(filters=width, name=f'{i}', **options)(outputs)
        else:
            outputs = layers.Conv2D(filters=width, name=f'{i}', **options)(outputs)
        outputs = layers.BatchNormalization(momentum=0.997, epsilon=1e-4, name=f'{i}-{level}')(outputs)
        outputs = layers.ReLU()(outputs)

    if detect_quadrangle:
        if separable_conv:
            outputs = layers.SeparableConv2D(num_anchors * 9, name=f'box-predict', **options)(outputs)
        else:
            outputs = layers.Conv2D(num_anchors * 9, **options)(outputs)
        # (b, num_anchors_this_feature_map, 9)
        outputs = layers.Reshape((-1, 9))(outputs)
    else:
        if separable_conv:
            outputs = layers.SeparableConv2D(num_anchors * 9, name=f'box-predict', **options)(outputs)
        else:
            outputs = layers.Conv2D(num_anchors * 4, **options)(outputs)
        # (b, num_anchors_this_feature_map, 4)
        outputs = layers.Reshape((-1, 4))(outputs)

    return outputs


class BoxNet(models.Model):
    def __init__(self, width, depth, num_anchors=9, separable_conv=True, **kwargs):
        super(BoxNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
            'bias_initializer': 'zeros',
        }
        if separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_anchors * 4, name=f'{self.name}/box-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, name=f'{self.name}/box-{i}', **options) for i in range(depth)]
            self.head = layers.Conv2D(filters=num_anchors * 4, name=f'{self.name}/box-predict', **options)
        self.bns = [[layers.BatchNormalization(momentum=0.997, epsilon=1e-4, name=f'{self.name}/box-{i}-bn-{j}')
                     for j in range(3, 8)] for i in range(depth)]
        # self.relu = layers.ReLU()
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, 4))

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            # feature = tf.nn.swish(feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        return outputs


class ClassNet(models.Model):
    def __init__(self, width, depth, num_classes=20, num_anchors=9, separable_conv=True, **kwargs):
        super(ClassNet, self).__init__(**kwargs)
        self.width = width
        self.depth = depth
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.separable_conv = separable_conv
        options = {
            'kernel_size': 3,
            'strides': 1,
            'padding': 'same',
        }
        if self.separable_conv:
            kernel_initializer = {
                'depthwise_initializer': initializers.VarianceScaling(),
                'pointwise_initializer': initializers.VarianceScaling(),
            }
            options.update(kernel_initializer)
            self.convs = [layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                                 **options)
                          for i in range(depth)]
            self.head = layers.SeparableConv2D(filters=num_classes * num_anchors,
                                               bias_initializer=PriorProbability(probability=0.01),
                                               name=f'{self.name}/class-predict', **options)
        else:
            kernel_initializer = {
                'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
            }
            options.update(kernel_initializer)
            self.convs = [layers.Conv2D(filters=width, bias_initializer='zeros', name=f'{self.name}/class-{i}',
                                        **options)
                          for i in range(depth)]
            self.head = layers.Conv2D(filters=num_classes * num_anchors,
                                      bias_initializer=PriorProbability(probability=0.01),
                                      name='class-predict', **options)
        self.bns = [[layers.BatchNormalization(momentum=0.997, epsilon=1e-4, name=f'{self.name}/class-{i}-bn-{j}')
                     for j in range(3, 8)] for i in range(depth)]
        # self.relu = layers.ReLU()
        self.relu = layers.Lambda(lambda x: tf.nn.swish(x))
        self.reshape = layers.Reshape((-1, num_classes))
        self.activation = layers.Activation('sigmoid')

    def call(self, inputs, **kwargs):
        feature, level = inputs
        for i in range(self.depth):
            feature = self.convs[i](feature)
            feature = self.bns[i][level](feature)
            # feature = tf.nn.swish(feature)
            feature = self.relu(feature)
        outputs = self.head(feature)
        outputs = self.reshape(outputs)
        outputs = self.activation(outputs)
        return outputs


def build_class_conv(width, depth, separable_conv=True):
    if separable_conv:
        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
    else:
        kernel_initializer = {
            'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }
    options.update(kernel_initializer)
    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        if separable_conv:
            outputs = layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'class-{i}', **options, )(
                outputs)
        else:
            outputs = layers.Conv2D(filters=width, bias_initializer='zeros', name=f'class-{i}', **options)(outputs)

    return models.Model(inputs=inputs, outputs=outputs, name='class_net_conv')


def build_class_head(width, num_classes=20, num_anchors=9, separable_conv=True):
    if separable_conv:
        kernel_initializer = {
            'depthwise_initializer': initializers.VarianceScaling(),
            'pointwise_initializer': initializers.VarianceScaling(),
        }
    else:
        kernel_initializer = {
            'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }

    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }
    options.update(kernel_initializer)
    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    if separable_conv:
        outputs = layers.SeparableConv2D(
            filters=num_classes * num_anchors,
            bias_initializer=PriorProbability(probability=0.01),
            name='class-predict',
            **options
        )(outputs)
    else:
        outputs = layers.Conv2D(
            filters=num_classes * num_anchors,
            bias_initializer=PriorProbability(probability=0.01),
            name='class-predict',
            **options
        )(outputs)
    # (b, num_anchors_this_feature_map, 4)
    outputs = layers.Reshape((-1, num_classes))(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    return models.Model(inputs=inputs, outputs=outputs, name='class_net')


def efficientdet(phi, num_classes=20, num_anchors=9, weighted_bifpn=False, freeze_bn=False,
                 score_threshold=0.01,
                 detect_quadrangle=False, anchor_parameters=None):
    assert phi in range(7)
    input_size = image_sizes[phi]
    input_shape = (input_size, input_size, 3)
    # input_shape = (None, None, 3)
    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = d_bifpns[phi]
    w_head = w_bifpn
    d_head = d_heads[phi]
    backbone_cls = backbones[phi]
    # features = backbone_cls(include_top=False, input_shape=input_shape, weights=weights)(image_input)
    features = backbone_cls(input_tensor=image_input, freeze_bn=freeze_bn)
    if weighted_bifpn:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_wBiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)
    else:
        fpn_features = features
        for i in range(d_bifpn):
            fpn_features = build_BiFPN(fpn_features, w_bifpn, i, freeze_bn=freeze_bn)
    box_net = BoxNet(w_head, d_head, num_anchors=num_anchors, name='box_net')
    class_net = ClassNet(w_head, d_head, num_classes=num_classes, num_anchors=num_anchors, name='class_net')
    classification = [class_net([feature, i]) for i, feature in enumerate(fpn_features)]
    classification = layers.Concatenate(axis=1, name='classification')(classification)
    regression = [box_net([feature, i]) for i, feature in enumerate(fpn_features)]
    regression = layers.Concatenate(axis=1, name='regression')(regression)

    # all_feats = [
    #     model.get_layer('block3c_add').output,
    #     model.get_layer('block5d_add').output,
    #     model.get_layer('block7b_add').output,
    #     model.get_layer('resample_p6/maxpool').output,
    #     model.get_layer('resample_p7/maxpool').output,
    #     model.get_layer('fpn_cells/cell_0/fnode3/op_after_combine8/bn').output,
    #     model.get_layer('fpn_cells/cell_0/fnode2/op_after_combine7/bn').output,
    #     model.get_layer('fpn_cells/cell_0/fnode1/op_after_combine6/bn').output,
    #     model.get_layer('fpn_cells/cell_0/fnode0/op_after_combine5/bn').output,
    #     model.get_layer('fpn_cells/cell_0/fnode7/op_after_combine12/bn').output,
    #     model.get_layer('fpn_cells/cell_1/fnode3/op_after_combine8/bn').output,
    #     model.get_layer('fpn_cells/cell_1/fnode2/op_after_combine7/bn').output,
    #     model.get_layer('fpn_cells/cell_1/fnode1/op_after_combine6/bn').output,
    #     model.get_layer('fpn_cells/cell_1/fnode0/op_after_combine5/bn').output,
    #     model.get_layer('fpn_cells/cell_1/fnode7/op_after_combine12/bn').output,
    #     model.get_layer('fpn_cells/cell_2/fnode3/op_after_combine8/bn').output,
    #     model.get_layer('fpn_cells/cell_2/fnode2/op_after_combine7/bn').output,
    #     model.get_layer('fpn_cells/cell_2/fnode1/op_after_combine6/bn').output,
    #     model.get_layer('fpn_cells/cell_2/fnode0/op_after_combine5/bn').output,
    #     model.get_layer('fpn_cells/cell_2/fnode7/op_after_combine12/bn').output,
    #     model.get_layer('fpn_cells/cell_3/fnode3/op_after_combine8/bn').output,
    #     model.get_layer('fpn_cells/cell_3/fnode2/op_after_combine7/bn').output,
    #     model.get_layer('fpn_cells/cell_3/fnode1/op_after_combine6/bn').output,
    #     model.get_layer('fpn_cells/cell_3/fnode0/op_after_combine5/bn').output,
    #     model.get_layer('fpn_cells/cell_3/fnode7/op_after_combine12/bn').output,
    # ]
    # model = models.Model(inputs=[image_input], outputs=[classification, regression, all_feats], name='efficientdet')
    model = models.Model(inputs=[image_input], outputs=[classification, regression], name='efficientdet')

    # apply predicted regression to anchors
    anchors = anchors_for_shape((input_size, input_size), anchor_params=anchor_parameters)
    anchors_input = np.expand_dims(anchors, axis=0)
    boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    if detect_quadrangle:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold,
            detect_quadrangle=True
        )([boxes, classification, regression[..., 4:8], regression[..., 8]])
    else:
        detections = FilterDetections(
            name='filtered_detections',
            score_threshold=score_threshold
        )([boxes, classification])

    prediction_model = models.Model(inputs=[image_input], outputs=detections, name='efficientdet_p')
    return model, prediction_model


def group_weights(weights):
    """
    Group each layer weights together, initially all weights are dict of 'layer_name/layer_var': np.array
    NOTE: weights 里面有网络最后的 weights 和 exponential moving average weights
    EMA 具体可参考 https://github.com/tensorflow/tpu/issues/408
    Example:
        input:  {
                    ...: ...
                    'conv2d/kernel': <np.array>,
                    'conv2d/bias': <np.array>,
                    ...: ...
                }
        output: [..., [...], [<conv2d/kernel-weights>, <conv2d/bias-weights>], [...], ...]

    """
    layer_weights = {}
    previous_layer_name = ""
    group = []

    for k, v in weights.items():
        layer_name = "/".join(k.split("/")[:-1])
        if layer_name == previous_layer_name:
            group.append(v)
        else:
            if group:
                # 把上一层的 weights group 放到 layer_weights
                layer_weights[previous_layer_name] = group
            # 新的一层的开始
            group = [v]
            previous_layer_name = layer_name
    # 收尾
    layer_weights[layer_name] = group
    return layer_weights


if __name__ == '__main__':
    from pprint import pprint
    import pickle

    model, _ = efficientdet(1, weighted_bifpn=True, num_classes=90)
    model.summary()
    # for layer in model.layers:
    #     print(layer, '\t', layer.name)
    # for weight in model.weights:
    #     pprint(weight.name)
    # print(cnt)
    tf_weights = pickle.load(
        open('/home/adam/github/others/automl/efficientdet/checkpoints/efficientdet-d1_weights.pkl', 'rb'))
    tf_layer_weights = group_weights(tf_weights)
    tf_layer_weights_keys = list(tf_layer_weights.keys())
    tf_layer_weights_values = list(tf_layer_weights.values())
    tf_index = 0
    index = 0
    for layer in model.layers:
        if tf_index == 182:
            break
        if isinstance(layer, (layers.Conv2D, layers.BatchNormalization, layers.SeparableConv2D)):
            print(layer, '\t', layer.name, '\t', tf_layer_weights_keys[tf_index])
            layer.set_weights(tf_layer_weights_values[tf_index])
            tf_index += 1
        index += 1
    for layer in model.layers[index:]:
        if isinstance(layer, (layers.Conv2D, layers.BatchNormalization, layers.SeparableConv2D, wBiFPNAdd)):
            if layer.name in tf_layer_weights:
                if layer.name[:-1].endswith('fnode'):
                    layer.set_weights([np.array(tf_layer_weights[layer.name])])
                else:
                    layer.set_weights(tf_layer_weights[layer.name])
            else:
                print(f'{layer.name} not found')
    model.save('d1.h5')
    # # for layer, tf_layer in zip(layer_weights[182:], list(tf_layer_weights.keys())[182:]):
    # #     if layer.name != tf_layer:
    # #         print(layer, '\t', layer.name, '\t', tf_layer)
    # print(set(list(tf_layer_weights.keys())[182:]) - set([layer.name for layer in layer_weights[182:]]))
    # # for x in (set(list(tf_layer_weights.keys())[182:]) - set([layer.name for layer in layer_weights[182:]])):
    # #     if x.startswith('fpn'):
    # #         print(x)
    # print(set([layer.name for layer in layer_weights[182:]]) - set(list(tf_layer_weights.keys())[182:]))
