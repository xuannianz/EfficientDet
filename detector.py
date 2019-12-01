from model import efficientdet
import os
import cv2
import numpy as np
from utils.anchors import anchors_for_shape, anchor_targets_bbox

import time

def generate_voc_classes():

    voc_classes = {
            0 : 'aeroplane',
            1 : 'bicycle',
            2 :'bird',
            3 : 'boat',
            4 : 'bottle',
            5 : 'bus',
            6 : 'car',
            7 : 'cat',
            8 : 'chair',
            9 : 'cow',
            10 : 'diningtable',
            11 : 'dog',
            12 : 'horse',
            13 : 'motorbike',
            14 : 'person',
            15 : 'pottedplant',
            16 : 'sheep',
            17 : 'sofa',
            18 : 'train',
            19 : 'tvmonitor'
            }
    return voc_classes
  
def generate_class_colors(num_classes):
    color_dict = dict()
    colors = []
    for index, i in enumerate(range(num_classes)):
        color_dict[index] = tuple(np.random.choice(range(256), size=3))

    return color_dict

def generate_resolutions():
    return [512, 640, 768, 896, 1024, 1280, 1408]
    
def preprocess_image(image, image_size=512):
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


def detect_on_frame(image, prediction_model, anchors, score_threshold=0.5, max_detections=100):
    h, w = image.shape[:2]
    image, scale, offset_h, offset_w = preprocess_image(image)

    # time to detect
    # run network
    boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                         np.expand_dims(anchors, axis=0)])
    boxes[..., [0, 2]] = boxes[..., [0, 2]] - offset_w
    boxes[..., [1, 3]] = boxes[..., [1, 3]] - offset_h
    boxes /= scale
    boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
    boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
    boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
    boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)

    # select indices which have a score above the threshold
    indices = np.where(scores[0, :] > score_threshold)[0]

    # select those scores
    scores = scores[0][indices]

    # find the order with which to sort the scores
    scores_sort = np.argsort(-scores)[:max_detections]

    # select detections
    # (n, 4)
    image_boxes = boxes[0, indices[scores_sort], :]
    # (n, )
    image_scores = scores[scores_sort]
    # (n, )
    image_labels = labels[0, indices[scores_sort]]
    # (n, 6)
    detections = np.concatenate(
        [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
    return detections


if __name__ == "__main__":
    model_path = 'model.h5'
    phi = 0
    object_classes = generate_voc_classes()
    resolutions = generate_resolutions()
    score_threshold = 0.5
    max_detections = 100
    image = cv2.imread("image.jpg")

    num_classes=len(object_classes)
    colors = generate_class_colors(num_classes)
    anchors = anchors_for_shape((resolutions[phi], resolutions[phi]))

    model, prediction_model = efficientdet(phi=phi, num_classes=num_classes)
    prediction_model.load_weights(model_path, by_name=True)

    draw_image = image.copy()
    start_time = time.time()
    detections = detect_on_frame(image, prediction_model, anchors, score_threshold, max_detections)
    print("Prediction speed {}".format(1/(time.time() - start_time)))

    ## Visualise
    for detection in detections:
        label = int(detection[5])
        color = colors[label]
        cv2.rectangle(draw_image, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (int(color[0]), int(color[1]), int(color[2])), 3)

    cv2.imshow("results", draw_image)
    cv2.waitKey(0)

