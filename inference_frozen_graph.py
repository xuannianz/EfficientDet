import tensorflow as tf
import numpy as np
import cv2
import os
import time
from utils import preprocess_image
from tensorflow.python.platform import gfile

phi = 1
weighted_bifpn = False
model_path = 'checkpoints/2019-12-03/pascal_05.pb'
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
num_classes = len(classes)
score_threshold = 0.5
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

tf.reset_default_graph()
sess = tf.Session()
with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

tensor_input = sess.graph.get_tensor_by_name('input_1:0')

output_boxes = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0')
output_scores = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0')
output_labels = sess.graph.get_tensor_by_name('filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0')


image_path = 'datasets/VOC2007/JPEGImages/000002.jpg'
image = cv2.imread(image_path)
src_image = image.copy()
image = image[:, :, ::-1]
h, w = image.shape[:2]

image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
# run network
start = time.time()
boxes, scores, labels = sess.run([output_boxes, output_scores, output_labels], {tensor_input:[image]})
print(time.time() - start)
boxes[0, :, [0, 2]] = boxes[0, :, [0, 2]] - offset_w
boxes[0, :, [1, 3]] = boxes[0, :, [1, 3]] - offset_h
boxes /= scale
boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)

# select indices which have a score above the threshold
indices = np.where(scores[0, :] > score_threshold)[0]

# select those detections
boxes = boxes[0, indices]
scores = scores[0, indices]
labels = labels[0, indices]

for box, score, label in zip(boxes, scores, labels):
    xmin = int(round(box[0]))
    ymin = int(round(box[1]))
    xmax = int(round(box[2]))
    ymax = int(round(box[3]))
    score = '{:.4f}'.format(score)
    class_id = int(label)
    color = colors[class_id]
    class_name = classes[class_id]
    label = '-'.join([class_name, score])
    ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), color, 1)
    cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
    cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', src_image)
cv2.waitKey(0)
