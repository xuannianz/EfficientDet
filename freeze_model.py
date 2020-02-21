from model import efficientdet
import cv2
import os
import numpy as np
import time
from utils import preprocess_image
import tensorflow as tf
from tensorflow.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def main():
    phi = 1
    weighted_bifpn = False
    model_path = 'checkpoints/2019-12-03/pascal_05_0.6283_1.1975_0.8029.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
    ]
    num_classes = len(classes)
    score_threshold = 0.5
    model, prediction_model = efficientdet(phi=phi,
                                           weighted_bifpn=weighted_bifpn,
                                           num_classes=num_classes,
                                           score_threshold=score_threshold)
    prediction_model.load_weights(model_path, by_name=True)
    
    frozen_graph = freeze_session(K.get_session(),  output_names=[out.op.name for out in prediction_model.outputs])
    tf.train.write_graph(frozen_graph, "./checkpoints/2019-12-03/", "pascal_05.pb", as_text=False)


if __name__ == '__main__':
    main()
