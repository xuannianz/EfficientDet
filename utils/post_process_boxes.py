import numpy as np


def post_process_boxes(boxes, scale, offset_h, offset_w, height, width):
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - offset_w
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - offset_h
    boxes /= scale
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    
    return boxes
