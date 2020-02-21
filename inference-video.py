import os
import cv2
import numpy as np

from model import efficientdet
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
        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor',
    ]
    num_classes = len(classes)
    score_threshold = 0.5
    colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
    model, prediction_model = efficientdet(phi=phi,
                                           weighted_bifpn=weighted_bifpn,
                                           num_classes=num_classes,
                                           score_threshold=score_threshold)
    prediction_model.load_weights(model_path, by_name=True)
    
    video_path = 'datasets/video.mp4'
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        image, scale, offset_h, offset_w = preprocess_image(frame, image_size=image_size)
        
        anchors = anchors_for_shape((image_size, image_size))
        
        boxes_batch, scores_batch, labels_batch = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                                                     np.expand_dims(anchors, axis=0)])
        
        for i, (boxes, scores, labels) in enumerate(zip(boxes_batch, scores_batch, labels_batch)):
            boxes = post_process_boxes(boxes=boxes,
                                       scale=scale,
                                       offset_h=offset_h,
                                       offset_w=offset_w,
                                       height=h,
                                       width=w)

            indices = np.where(scores[:] > score_threshold)[0]
            boxes = boxes[indices]
            labels = labels[indices]
            
            draw_boxes(frame, boxes, scores, labels, colors, classes)
            
            cv2.imshow('image', frame)
            cv2.waitKey(1)
            

if __name__ == '__main__':
    main()
