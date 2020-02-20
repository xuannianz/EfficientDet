from model import efficientdet
import cv2
import os
import numpy as np
import time
from utils import preprocess_image
from utils.anchors import anchors_for_shape
from utils.draw_boxes import draw_boxes
from utils.post_process_boxes import post_process_boxes


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
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
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
    model, prediction_model = efficientdet(phi=phi,
                                           weighted_bifpn=weighted_bifpn,
                                           num_classes=num_classes,
                                           score_threshold=score_threshold)
    prediction_model.load_weights(model_path, by_name=True)
    
    image_path = 'datasets/VOC2007/JPEGImages/000002.jpg'
    image = cv2.imread(image_path)
    src_image = image.copy()
    image = image[:, :, ::-1]
    h, w = image.shape[:2]
    
    image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
    inputs = np.expand_dims(image, axis=0)
    anchors = anchors_for_shape((image_size, image_size))
    # run network
    start = time.time()
    boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                               np.expand_dims(anchors, axis=0)])
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
