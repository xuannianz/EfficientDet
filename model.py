# from keras import layers
# from keras import initializers
# from keras import models
# from keras_ import EfficientNetB0, EfficientNetB1, EfficientNetB2
# from keras_ import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras import models
from tfkeras import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tfkeras import EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6

from layers import ClipBoxes, RegressBoxes, FilterDetections
from initializers import PriorProbability

w_bifpns = [64, 88, 112, 160, 224, 288, 384]
backbones = [EfficientNetB0, EfficientNetB1, EfficientNetB2,
             EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6]


def build_BiFPN(features, num_channels, id):
    if id == 0:
        _, _, C3, C4, C5 = features
        P3 = layers.Conv2D(num_channels, kernel_size=1, strides=1, padding='same', name='BiFPN_{}_P3'.format(id))(C3)
        P4 = layers.Conv2D(num_channels, kernel_size=1, strides=1, padding='same', name='BiFPN_{}_P4'.format(id))(C4)
        P5 = layers.Conv2D(num_channels, kernel_size=1, strides=1, padding='same', name='BiFPN_{}_P5'.format(id))(C5)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = layers.Conv2D(num_channels, kernel_size=3, strides=2, padding='same', name='BiFPN_{}_P6'.format(id))(C5)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = layers.Activation('relu', name='C6_relu')(P6)
        P7 = layers.Conv2D(num_channels, kernel_size=3, strides=2, padding='same', name='BiFPN_{}_P7'.format(id))(P7)
    else:
        P3, P4, P5, P6, P7 = features

    # upsample
    P7_U = layers.UpSampling2D()(P7)
    P6 = layers.Add()([P7_U, P6])
    P6_U = layers.UpSampling2D()(P6)
    P5 = layers.Add()([P6_U, P5])
    P5_U = layers.UpSampling2D()(P5)
    P4 = layers.Add()([P5_U, P4])
    P4_U = layers.UpSampling2D()(P4)
    P3 = layers.Add()([P4_U, P3])
    P3 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_U_P3'.format(id))(P3)
    P4 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_U_P4'.format(id))(P4)
    P5 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_U_P5'.format(id))(P5)
    P6 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_U_P6'.format(id))(P6)
    P7 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_U_P7'.format(id))(P7)
    # downsample
    P3_D = layers.MaxPooling2D(strides=(2, 2))(P3)
    P4 = layers.Add()([P3_D, P4])
    P4_D = layers.MaxPooling2D(strides=(2, 2))(P4)
    P5 = layers.Add()([P4_D, P5])
    P5_D = layers.MaxPooling2D(strides=(2, 2))(P5)
    P6 = layers.Add()([P5_D, P6])
    P6_D = layers.MaxPooling2D(strides=(2, 2))(P6)
    P7 = layers.Add()([P6_D, P7])
    P3 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_D_P3'.format(id))(P3)
    P4 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_D_P4'.format(id))(P4)
    P5 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_D_P5'.format(id))(P5)
    P6 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_D_P6'.format(id))(P6)
    P7 = layers.Conv2D(num_channels, kernel_size=3, strides=1, padding='same', name='BiFPN_{}_D_P7'.format(id))(P7)

    return P3, P4, P5, P6, P7


def build_regress_head(width, depth, num_anchors=9):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        # 'kernel_initializer': initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'kernel_initializer': initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        outputs = layers.Conv2D(
            filters=width,
            activation='relu',
            **options
        )(outputs)

    outputs = layers.Conv2D(num_anchors * 4, **options)(outputs)
    # (b, num_anchors_this_feature_map, 4)
    outputs = layers.Reshape((-1, 4))(outputs)

    return models.Model(inputs=inputs, outputs=outputs, name='box_head')


def build_class_head(width, depth, num_classes=20, num_anchors=9):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        # 'kernel_initializer': initializers.normal(mean=0.0, stddev=0.01, seed=None),
    }

    inputs = layers.Input(shape=(None, None, width))
    outputs = inputs
    for i in range(depth):
        outputs = layers.Conv2D(
            filters=width,
            activation='relu',
            kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    # outputs = layers.Conv2D(num_anchors * num_classes, **options)(outputs)
    outputs = layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=0.01),
        name='pyramid_classification',
        **options
    )(outputs)
    # (b, num_anchors_this_feature_map, 4)
    outputs = layers.Reshape((-1, num_classes))(outputs)
    outputs = layers.Activation('sigmoid')(outputs)

    return models.Model(inputs=inputs, outputs=outputs, name='class_head')


def efficientdet(phi, num_classes=20):
    assert phi in range(7)
    input_size = 512 + phi * 128
    input_shape = (input_size, input_size, 3)
    # input_shape = (None, None, 3)
    image_input = layers.Input(input_shape)
    w_bifpn = w_bifpns[phi]
    d_bifpn = 2 + phi
    w_head = w_bifpn
    d_head = 3 + int(phi / 3)
    backbone_cls = backbones[phi]
    # features = backbone_cls(include_top=False, input_shape=input_shape, weights=weights)(image_input)
    features = backbone_cls(input_tensor=image_input)
    for i in range(d_bifpn):
        features = build_BiFPN(features, w_bifpn, i)
    regress_head = build_regress_head(w_head, d_head)
    class_head = build_class_head(w_head, d_head, num_classes=num_classes)
    regression = [regress_head(feature) for feature in features]
    regression = layers.Concatenate(axis=1, name='regression')(regression)
    classification = [class_head(feature) for feature in features]
    classification = layers.Concatenate(axis=1, name='classification')(classification)

    model = models.Model(inputs=[image_input], outputs=[regression, classification], name='efficientnet')

    # apply predicted regression to anchors
    # anchors = tf.tile(tf.expand_dims(tf.constant(anchors), axis=0), (tf.shape(regression)[0], 1, 1))
    anchors_input = layers.Input((None, 4))
    boxes = RegressBoxes(name='boxes')([anchors_input, regression])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        name='filtered_detections'
    )([boxes, classification])
    prediction_model = models.Model(inputs=[image_input, anchors_input], outputs=detections, name='efficientnet_p')
    return model, prediction_model


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    model, prediction_model = efficientdet(phi=0, num_classes=20)
    model.layers[1].trainable = False
    # for layer in model.layers[1].layers:
    #     layer.trainable = False
    for i, layer in enumerate(model.layers):
        print(i, '\t', layer, '\t\t', layer.name, '\t', layer.trainable)
        if layer.__class__.__name__ == 'Model':
            for j, layer_ in enumerate(layer.layers):
                print('\t', j, '\t', layer_, '\t\t', layer_.name, '\t', layer_.trainable)
