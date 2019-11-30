from model import efficientdet
import os
import cv2
import numpy as np
from utils.anchors import anchors_for_shape, anchor_targets_bbox

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



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model_path = 'model.h5'
    phi = 0
    num_classes=20
    score_threshold = 0.5
    max_detections = 100
    image = cv2.imread("image.jpg")

    anchors = anchors_for_shape((512, 512))

    model, prediction_model = efficientdet(phi=phi, num_classes=num_classes)
    prediction_model.load_weights(model_path, by_name=True)

    h, w = image.shape[:2]
    draw_image = image.copy()
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
    print(detections)



## Visualise
    for detection in detections:
        print("------------------")
        print(detection)
        label = int(detection[5])
        if label != 14:
            print("not person")
            continue

        print((detection[0], detection[1]))
        print((detection[2], detection[3]))
        cv2.rectangle(draw_image, (int(detection[0]), int(detection[1])), (int(detection[2]), int(detection[3])), (255,255,255), 3)
        print("------------------")

    cv2.imshow("results", draw_image)
    cv2.waitKey(0)
    #draw_detections(src_image, detections[:5, :4], detections[:5, 4], detections[:5, 5].astype(np.int32),
    #    label_to_name=generator.label_to_name,
    #    score_threshold=score_threshold)

    # cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
    #cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
    #cv2.imshow('{}'.format(i), src_image)
    #cv2.waitKey(0)

