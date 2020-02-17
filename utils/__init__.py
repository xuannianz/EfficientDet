# Copyright 2019 The TensorFlow Authors, Pavel Yakubovskiy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import functools
import cv2
import numpy as np

_KERAS_BACKEND = None
_KERAS_LAYERS = None
_KERAS_MODELS = None
_KERAS_UTILS = None


def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', _KERAS_BACKEND)
    layers = kwargs.get('layers', _KERAS_LAYERS)
    models = kwargs.get('models', _KERAS_MODELS)
    utils = kwargs.get('utils', _KERAS_UTILS)
    for key in kwargs.keys():
        if key not in ['backend', 'layers', 'models', 'utils']:
            raise TypeError('Invalid keyword argument: %s', key)
    return backend, layers, models, utils


def inject_keras_modules(func):
    import keras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = keras.backend
        kwargs['layers'] = keras.layers
        kwargs['models'] = keras.models
        kwargs['utils'] = keras.utils
        return func(*args, **kwargs)

    return wrapper


def inject_tfkeras_modules(func):
    import tensorflow.keras as tfkeras
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs['backend'] = tfkeras.backend
        kwargs['layers'] = tfkeras.layers
        kwargs['models'] = tfkeras.models
        kwargs['utils'] = tfkeras.utils
        return func(*args, **kwargs)

    return wrapper


def init_keras_custom_objects():
    import keras
    import efficientnet as model

    custom_objects = {
        'swish': inject_keras_modules(model.get_swish)(),
        'FixedDropout': inject_keras_modules(model.get_dropout)()
    }

    keras.utils.generic_utils.get_custom_objects().update(custom_objects)


def init_tfkeras_custom_objects():
    import tensorflow.keras as tfkeras
    import efficientnet as model

    custom_objects = {
        'swish': inject_tfkeras_modules(model.get_swish)(),
        'FixedDropout': inject_tfkeras_modules(model.get_dropout)()
    }

    tfkeras.utils.get_custom_objects().update(custom_objects)


def preprocess_image(image, image_size):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size
    image = cv2.resize(image, (resized_width, resized_height))
    new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
    offset_h = (image_size - resized_height) // 2
    offset_w = (image_size - resized_width) // 2
    new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
    new_image /= 255.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_image[..., 0] -= mean[0]
    new_image[..., 1] -= mean[1]
    new_image[..., 2] -= mean[2]
    new_image[..., 0] /= std[0]
    new_image[..., 1] /= std[1]
    new_image[..., 2] /= std[2]
    return new_image, scale, offset_h, offset_w


def rotate_image(image):
    rotate_degree = np.random.uniform(low=-45, high=45)
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(128, 128, 128))

    return image
