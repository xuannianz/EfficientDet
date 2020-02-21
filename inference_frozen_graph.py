import tensorflow as tf
import numpy as np
import cv2
import os
import time
from utils import preprocess_image
from tensorflow.python.platform import gfile

from utils.anchors import anchors_for_shape
from utils.draw_boxes import draw_boxes
from utils.post_process_boxes import post_process_boxes


def get_frozen_graph(graph_file):
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def main():
    phi = 1
    model_path = 'checkpoints/2019-12-03/pascal_05.pb'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    classes = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor',
    ]
    num_classes = len(classes)
    score_threshold = 0.5
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

    output_names = {
        'output_boxes': 'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',
        'output_scores': 'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',
        'output_labels': 'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0'
    }

    graph = tf.Graph()
    graph.as_default()
    sess = tf.Session()
    graph = get_frozen_graph(model_path)
    tf.import_graph_def(graph, name='')

    output_boxes = sess.graph.get_tensor_by_name(output_names["output_boxes"])
    output_scores = sess.graph.get_tensor_by_name(output_names['output_scores'])
    output_labels = sess.graph.get_tensor_by_name(output_names['output_labels'])
    
    image_path = 'datasets/VOC2007/JPEGImages/000002.jpg'
    image = cv2.imread(image_path)
    src_image = image.copy()
    image = image[:, :, ::-1]
    h, w = image.shape[:2]
    
    image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
    anchors = anchors_for_shape((image_size, image_size))
    
    # run network
    start = time.time()
    image_batch = np.expand_dims(image, axis=0)
    anchors_batch = np.expand_dims(anchors, axis=0)
    feed_dict = {"input_1:0": image_batch, "input_4:0": anchors_batch}
    boxes, scores, labels = sess.run([output_boxes, output_scores, output_labels], feed_dict)

    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    print(time.time() - start)
    boxes = post_process_boxes(boxes=boxes,
                               scale=scale,
                               offset_h=offset_h,
                               offset_w=offset_w,
                               height=h,
                               width=w)

    # select indices which have a score above the threshold
    indices = np.where(scores[:] > score_threshold)[0]

    # select those detections
    boxes = boxes[indices]
    labels = labels[indices]

    draw_boxes(src_image, boxes, scores, labels, colors, classes)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', src_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
