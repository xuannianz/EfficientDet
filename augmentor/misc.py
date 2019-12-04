import cv2
import numpy as np
from augmentor.transform import translation_xy, change_transform_origin, scaling_xy

ROTATE_DEGREE = [90, 180, 270]


def rotate(image, boxes, prob=0.5, border_value=(128, 128, 128)):
    boxes = boxes.astype(np.float32)
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    rotate_degree = ROTATE_DEGREE[np.random.randint(0, 3)]
    h, w = image.shape[:2]
    # Compute the rotation matrix.
    M = cv2.getRotationMatrix2D(center=(w / 2, h / 2),
                                angle=rotate_degree,
                                scale=1)

    # Get the sine and cosine from the rotation matrix.
    abs_cos_angle = np.abs(M[0, 0])
    abs_sin_angle = np.abs(M[0, 1])

    # Compute the new bounding dimensions of the image.
    new_w = int(h * abs_sin_angle + w * abs_cos_angle)
    new_h = int(h * abs_cos_angle + w * abs_sin_angle)

    # Adjust the rotation matrix to take into account the translation.
    M[0, 2] += new_w // 2 - w // 2
    M[1, 2] += new_h // 2 - h // 2

    # Rotate the image.
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=border_value)

    if boxes.shape[0] != 0:
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            points = M.dot([
                [x1, x2, x1, x2],
                [y1, y2, y2, y1],
                [1, 1, 1, 1],
            ])

            # Extract the min and max corners again.
            min_xy = np.sort(points, axis=1)[:, :2]
            min_x = np.mean(min_xy[0])
            min_y = np.mean(min_xy[1])
            max_xy = np.sort(points, axis=1)[:, 2:]
            max_x = np.mean(max_xy[0])
            max_y = np.mean(max_xy[1])

            new_boxes.append([min_x, min_y, max_x, max_y])
        boxes = np.array(new_boxes).astype(np.float32)
    return image, boxes


def crop(image, boxes, prob=0.5):
    boxes = boxes.astype(np.float32)
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    h, w = image.shape[:2]
    if boxes.shape[0] != 0:
        min_x1, min_y1 = np.min(boxes, axis=0)[:2]
        max_x2, max_y2 = np.max(boxes, axis=0)[2:]
        random_x1 = np.random.randint(0, max(min_x1 // 2, 1))
        random_y1 = np.random.randint(0, max(min_y1 // 2, 1))
        random_x2 = np.random.randint(max_x2, max(min(w, max_x2 + (w - max_x2) // 2), max_x2 + 1))
        random_y2 = np.random.randint(max_y2, max(min(h, max_y2 + (h - max_y2) // 2), max_y2 + 1))
        image = image[random_y1:random_y2, random_x1:random_x2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] - random_x1
        boxes[:, [1, 3]] = boxes[:, [1, 3]] - random_y1
        boxes = boxes.astype(np.float32)
    else:
        random_x1 = np.random.randint(0, max(w // 8, 1))
        random_y1 = np.random.randint(0, max(h // 8, 1))
        random_x2 = np.random.randint(7 * w // 8, w - 1)
        random_y2 = np.random.randint(7 * h // 8, h - 1)
        image = image[random_y1:random_y2, random_x1:random_x2]
    return image, boxes


def flipx(image, boxes, prob=0.5):
    boxes = boxes.astype(np.float32)
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    image = image[:, ::-1]
    h, w = image.shape[:2]
    if boxes.shape[0] != 0:
        tmp = boxes[:, 0].copy()
        boxes[:, 0] = w - boxes[:, 2]
        boxes[:, 2] = w - tmp
        boxes = boxes.astype(np.float32)
    return image, boxes


def multi_scale(image, boxes, prob=1.):
    boxes = boxes.astype(np.float32)
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    h, w = image.shape[:2]
    scale = np.random.choice(np.arange(0.7, 1.4, 0.1))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if boxes.shape[0] != 0:
        boxes = np.round(boxes * scale).astype(np.float32)
    return image, boxes


def translate(image, boxes, prob=0.5, border_value=(128, 128, 128)):
    boxes = boxes.astype(np.float32)
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, boxes
    h, w = image.shape[:2]
    if boxes.shape[0] != 0:
        min_x1, min_y1 = np.min(boxes, axis=0)[:2]
        max_x2, max_y2 = np.max(boxes, axis=0)[2:]
        translation_matrix = translation_xy(min=(min(-min_x1 // 2, 0), min(-min_y1 // 2, 0)),
                                            max=(max((w - max_x2) // 2, 1), max((h - max_y2) // 2, 1)), prob=1.)
    else:
        translation_matrix = translation_xy(min=(min(-w // 8, 0), min(-h // 8, 0)),
                                            max=(max(w // 8, 1), max(h // 8, 1)))
    translation_matrix = change_transform_origin(translation_matrix, (w / 2, h / 2))
    image = cv2.warpAffine(
        image,
        translation_matrix[:2, :],
        dsize=(w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    if boxes.shape[0] != 0:
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            points = translation_matrix.dot([
                [x1, x2, x1, x2],
                [y1, y2, y2, y1],
                [1, 1, 1, 1],
            ])
            min_x, min_y = np.min(points, axis=1)[:2]
            max_x, max_y = np.max(points, axis=1)[:2]
            new_boxes.append([min_x, min_y, max_x, max_y])
        boxes = np.array(new_boxes).astype(np.float32)
    return image, boxes


class MiscEffect:
    def __init__(self, multi_scale_prob=0.5, rotate_prob=0.05, flip_prob=0.5, crop_prob=0.5, translate_prob=0.5,
                 border_value=(128, 128, 128)):
        self.multi_scale_prob = multi_scale_prob
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.translate_prob = translate_prob
        self.border_value = border_value

    def __call__(self, image, boxes):
        image, boxes = multi_scale(image, boxes, prob=self.multi_scale_prob)
        image, boxes = rotate(image, boxes, prob=self.rotate_prob, border_value=self.border_value)
        image, boxes = flipx(image, boxes, prob=self.flip_prob)
        image, boxes = crop(image, boxes, prob=self.crop_prob)
        image, boxes = translate(image, boxes, prob=self.translate_prob, border_value=self.border_value)
        return image, boxes


if __name__ == '__main__':
    # from generators.pascal import PascalVocGenerator
    #
    # train_generator = PascalVocGenerator(
    #     'datasets/VOC0712',
    #     'trainval',
    #     skip_difficult=True,
    #     batch_size=1,
    #     shuffle_groups=False
    # )
    from generators.coco import CocoGenerator

    train_generator = CocoGenerator(
        '/home/adam/.keras/datasets/coco/2017_118_5',
        'train2017',
        batch_size=1,
        shuffle_groups=False
    )
    misc_effect = MiscEffect()
    for i in range(train_generator.size()):
        image = train_generator.load_image(i)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotations = train_generator.load_annotations(i)
        boxes = annotations['bboxes']
        for box in boxes.astype(np.int32):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        src_image = image.copy()
        # cv2.namedWindow('src_image', cv2.WINDOW_NORMAL)
        cv2.imshow('src_image', src_image)
        image, boxes = misc_effect(image, boxes)
        # image, boxes = multi_scale(image, boxes)
        image = image.copy()
        for box in boxes.astype(np.int32):
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        cv2.waitKey(0)
