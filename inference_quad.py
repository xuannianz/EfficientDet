from model import efficientdet
import cv2
import os
import numpy as np
import time
from utils import preprocess_image
from utils.anchors import anchors_for_shape, AnchorParameters
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

phi = 3
weighted_bifpn = False
model_path = 'checkpoints/2020-02-20/csv_02_1.6506_2.5878_w.h5'
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi]
# classes = [
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
# ]
classes = ['text']
num_classes = len(classes)
anchor_parameters = AnchorParameters(
    ratios=(0.25, 0.5, 1., 2.),
    sizes=(16, 32, 64, 128, 256))
score_threshold = 0.4
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       num_anchors=anchor_parameters.num_anchors(),
                                       score_threshold=score_threshold,
                                       detect_quadrangle=True,
                                       anchor_parameters=anchor_parameters,
                                       )
prediction_model.load_weights(model_path, by_name=True)

import glob

for image_path in glob.glob('datasets/ic15/test_images/*.jpg'):
    image = cv2.imread(image_path)
    src_image = image.copy()
    image = image[:, :, ::-1]
    h, w = image.shape[:2]

    image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
    inputs = np.expand_dims(image, axis=0)
    anchors = anchors_for_shape((image_size, image_size), anchor_params=anchor_parameters)
    # run network
    start = time.time()
    boxes, scores, alphas, ratios, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                                               np.expand_dims(anchors, axis=0)])
    # alphas = np.exp(alphas)
    alphas = 1 / (1 + np.exp(-alphas))
    ratios = 1 / (1 + np.exp(-ratios))
    quadrangles = np.zeros(boxes.shape[:2] + (8,))
    quadrangles[:, :, 0] = boxes[:, :, 0] + (boxes[:, :, 2] - boxes[:, :, 0]) * alphas[:, :, 0]
    quadrangles[:, :, 1] = boxes[:, :, 1]
    quadrangles[:, :, 2] = boxes[:, :, 2]
    quadrangles[:, :, 3] = boxes[:, :, 1] + (boxes[:, :, 3] - boxes[:, :, 1]) * alphas[:, :, 1]
    quadrangles[:, :, 4] = boxes[:, :, 2] - (boxes[:, :, 2] - boxes[:, :, 0]) * alphas[:, :, 2]
    quadrangles[:, :, 5] = boxes[:, :, 3]
    quadrangles[:, :, 6] = boxes[:, :, 0]
    quadrangles[:, :, 7] = boxes[:, :, 3] - (boxes[:, :, 3] - boxes[:, :, 1]) * alphas[:, :, 3]
    print(time.time() - start)

    boxes[0, :, [0, 2]] = boxes[0, :, [0, 2]] - offset_w
    boxes[0, :, [1, 3]] = boxes[0, :, [1, 3]] - offset_h
    boxes /= scale
    boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
    boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
    boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
    boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)

    quadrangles[0, :, [0, 2, 4, 6]] = quadrangles[0, :, [0, 2, 4, 6]] - offset_w
    quadrangles[0, :, [1, 3, 5, 7]] = quadrangles[0, :, [1, 3, 5, 7]] - offset_h
    quadrangles /= scale
    quadrangles[0, :, [0, 2, 4, 6]] = np.clip(quadrangles[0, :, [0, 2, 4, 6]], 0, w - 1)
    quadrangles[0, :, [1, 3, 5, 7]] = np.clip(quadrangles[0, :, [1, 3, 5, 7]], 0, h - 1)

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those detections
    boxes = boxes[0, indices]
    scores = scores[0, indices]
    labels = labels[0, indices]
    quadrangles = quadrangles[0, indices]
    ratios = ratios[0, indices]

    for bbox, score, label, quadrangle, ratio in zip(boxes, scores, labels, quadrangles, ratios):
        xmin = int(round(bbox[0]))
        ymin = int(round(bbox[1]))
        xmax = int(round(bbox[2]))
        ymax = int(round(bbox[3]))
        score = '{:.4f}'.format(score)
        class_id = int(label)
        color = colors[class_id]
        class_name = classes[class_id]
        label = '-'.join([class_name, score])
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(src_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        # cv2.rectangle(src_image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        # cv2.putText(src_image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(src_image, f'{ratio:.2f}', (xmin + (xmax - xmin) // 3, (ymin + ymax) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.drawContours(src_image, [quadrangle.astype(np.int32).reshape((4, 2))], -1, (0, 0, 255), 1)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', src_image)
    cv2.waitKey(0)
