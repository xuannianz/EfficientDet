from generators.pascal import PascalVocGenerator
from model import efficientdet
import cv2
import os
import numpy as np
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
phi = 1
weighted_bifpn = False

generator = PascalVocGenerator(
    'datasets/VOC2007',
    'test',
    phi=phi,
    shuffle_groups=False,
    skip_truncated=False,
    skip_difficult=True,
)
model_path = 'checkpoints/2019-12-03/pascal_05_0.6283_1.1975_0.8029.h5'
num_classes = generator.num_classes()
classes = list(generator.classes.keys())
score_threshold = 0.5
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       score_threshold=score_threshold)
prediction_model.load_weights(model_path, by_name=True)
for i in range(10):
    image = generator.load_image(i)
    h, w = image.shape[:2]
    src_image = image[:, :, ::-1].copy()

    image, scale, offset_h, offset_w = generator.preprocess_image(image)
    inputs = np.expand_dims(image, axis=0)
    anchors = generator.anchors
    # run network
    start = time.time()
    boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                    np.expand_dims(anchors, axis=0)])
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
    key = cv2.waitKey(0)
    if int(key) == 121:
        image_fname = generator.image_names[i]
        cv2.imwrite('test/{}.jpg'.format(image_fname), src_image)
