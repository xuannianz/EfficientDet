import cv2
import numpy as np
from augmentor.transform import translation_xy, change_transform_origin, scaling_xy
from utils import reorder_vertexes


def rotate(image, annotations, prob=0.5, border_value=(128, 128, 128)):
    assert 'bboxes' in annotations, 'annotations should contain bboxes even if it is empty'

    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, annotations

    rotate_degree = np.random.uniform(low=-10, high=10)
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
    image = cv2.warpAffine(image, M=M, dsize=(new_w, new_h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=border_value)

    bboxes = annotations['bboxes']
    if bboxes.shape[0] != 0:
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
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
            new_bboxes.append([min_x, min_y, max_x, max_y])
        annotations['bboxes'] = np.array(new_bboxes, dtype=np.float32)

        if 'quadrangles' in annotations and annotations['quadrangles'].shape[0] != 0:
            quadrangles = annotations['quadrangles']
            rotated_quadrangles = []
            for quadrangle in quadrangles:
                quadrangle = np.concatenate([quadrangle, np.ones((4, 1))], axis=-1)
                rotated_quadrangle = M.dot(quadrangle.T).T[:, :2]
                quadrangle = reorder_vertexes(rotated_quadrangle)
                rotated_quadrangles.append(quadrangle)
            quadrangles = np.stack(rotated_quadrangles)
            annotations['quadrangles'] = quadrangles
            xmin = np.min(quadrangles, axis=1)[:, 0]
            ymin = np.min(quadrangles, axis=1)[:, 1]
            xmax = np.max(quadrangles, axis=1)[:, 0]
            ymax = np.max(quadrangles, axis=1)[:, 1]
            bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1)
            annotations['bboxes'] = bboxes
    return image, annotations


def crop(image, annotations, prob=0.5):
    assert 'bboxes' in annotations, 'annotations should contain bboxes even if it is empty'

    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, annotations
    h, w = image.shape[:2]
    bboxes = annotations['bboxes']
    if bboxes.shape[0] != 0:
        min_x1, min_y1 = np.min(bboxes, axis=0)[:2]
        max_x2, max_y2 = np.max(bboxes, axis=0)[2:]
        random_x1 = np.random.randint(0, max(min_x1 // 2, 1))
        random_y1 = np.random.randint(0, max(min_y1 // 2, 1))
        random_x2 = np.random.randint(max_x2 + 1, max(min(w, max_x2 + (w - max_x2) // 2), max_x2 + 2))
        random_y2 = np.random.randint(max_y2 + 1, max(min(h, max_y2 + (h - max_y2) // 2), max_y2 + 2))
        image = image[random_y1:random_y2, random_x1:random_x2]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - random_x1
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - random_y1
        if 'quadrangles' in annotations and annotations['quadrangles'].shape[0] != 0:
            quadrangles = annotations['quadrangles']
            quadrangles[:, :, 0] = quadrangles[:, :, 0] - random_x1
            quadrangles[:, :, 1] = quadrangles[:, :, 1] - random_y1
    else:
        random_x1 = np.random.randint(0, max(w // 8, 1))
        random_y1 = np.random.randint(0, max(h // 8, 1))
        random_x2 = np.random.randint(7 * w // 8, w - 1)
        random_y2 = np.random.randint(7 * h // 8, h - 1)
        image = image[random_y1:random_y2, random_x1:random_x2]
    return image, annotations


def flipx(image, annotations, prob=0.5):
    assert 'bboxes' in annotations, 'annotations should contain bboxes even if it is empty'

    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, annotations
    bboxes = annotations['bboxes']
    h, w = image.shape[:2]
    image = image[:, ::-1]
    if bboxes.shape[0] != 0:
        tmp = bboxes.copy()
        bboxes[:, 0] = w - 1 - bboxes[:, 2]
        bboxes[:, 2] = w - 1 - tmp[:, 0]
        if 'quadrangles' in annotations and annotations['quadrangles'].shape[0] != 0:
            quadrangles = annotations['quadrangles']
            tmp = quadrangles.copy()
            quadrangles[:, 0, 0] = w - 1 - quadrangles[:, 0, 0]
            quadrangles[:, 1, 0] = w - 1 - tmp[:, 3, 0]
            quadrangles[:, 1, 1] = tmp[:, 3, 1]
            quadrangles[:, 2, 0] = w - 1 - quadrangles[:, 2, 0]
            quadrangles[:, 3, 0] = w - 1 - tmp[:, 1, 0]
            quadrangles[:, 3, 1] = tmp[:, 1, 1]
    return image, annotations


def multi_scale(image, annotations, prob=1.):
    assert 'bboxes' in annotations, 'annotations should contain bboxes even if it is empty'

    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, annotations
    h, w = image.shape[:2]
    scale = np.random.choice(np.arange(0.7, 1.4, 0.1))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    bboxes = annotations['bboxes']
    if bboxes.shape[0] != 0:
        annotations['bboxes'] = np.round(bboxes * scale)
        if 'quadrangles' in annotations and annotations['quadrangles'].shape[0] != 0:
            quadrangles = annotations['quadrangles']
            annotations['quadrangles'] = np.round(quadrangles * scale)
    return image, annotations


def translate(image, annotations, prob=0.5, border_value=(128, 128, 128)):
    assert 'bboxes' in annotations, 'annotations should contain bboxes even if it is empty'

    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, annotations
    h, w = image.shape[:2]
    bboxes = annotations['bboxes']
    if bboxes.shape[0] != 0:
        min_x1, min_y1 = np.min(bboxes, axis=0)[:2].astype(np.int32)
        max_x2, max_y2 = np.max(bboxes, axis=0)[2:].astype(np.int32)
        translation_matrix = translation_xy(min=(min(-(min_x1 // 2), 0), min(-(min_y1 // 2), 0)),
                                            max=(max((w - 1 - max_x2) // 2, 1), max((h - 1 - max_y2) // 2, 1)),
                                            prob=1.)
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
    if bboxes.shape[0] != 0:
        new_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            points = translation_matrix.dot([
                [x1, x2, x1, x2],
                [y1, y2, y2, y1],
                [1, 1, 1, 1],
            ])
            min_x, min_y = np.min(points, axis=1)[:2]
            max_x, max_y = np.max(points, axis=1)[:2]
            new_bboxes.append([min_x, min_y, max_x, max_y])
        annotations['bboxes'] = np.array(new_bboxes).astype(np.float32)

        if 'quadrangles' in annotations and annotations['quadrangles'].shape[0] != 0:
            quadrangles = annotations['quadrangles']
            translated_quadrangles = []
            for quadrangle in quadrangles:
                quadrangle = np.concatenate([quadrangle, np.ones((4, 1))], axis=-1)
                translated_quadrangle = translation_matrix.dot(quadrangle.T).T[:, :2]
                quadrangle = reorder_vertexes(translated_quadrangle)
                translated_quadrangles.append(quadrangle)
            quadrangles = np.stack(translated_quadrangles)
            annotations['quadrangles'] = quadrangles
            xmin = np.min(quadrangles, axis=1)[:, 0]
            ymin = np.min(quadrangles, axis=1)[:, 1]
            xmax = np.max(quadrangles, axis=1)[:, 0]
            ymax = np.max(quadrangles, axis=1)[:, 1]
            bboxes = np.stack([xmin, ymin, xmax, ymax], axis=1)
            annotations['bboxes'] = bboxes

    return image, annotations


class MiscEffect:
    def __init__(self, multi_scale_prob=0.5, rotate_prob=0.05, flip_prob=0.5, crop_prob=0.5, translate_prob=0.5,
                 border_value=(128, 128, 128)):
        self.multi_scale_prob = multi_scale_prob
        self.rotate_prob = rotate_prob
        self.flip_prob = flip_prob
        self.crop_prob = crop_prob
        self.translate_prob = translate_prob
        self.border_value = border_value

    def __call__(self, image, annotations):
        image, annotations = multi_scale(image, annotations, prob=self.multi_scale_prob)
        image, annotations = rotate(image, annotations, prob=self.rotate_prob, border_value=self.border_value)
        image, annotations = flipx(image, annotations, prob=self.flip_prob)
        image, annotations = crop(image, annotations, prob=self.crop_prob)
        image, annotations = translate(image, annotations, prob=self.translate_prob, border_value=self.border_value)
        return image, annotations


if __name__ == '__main__':
    from generators.csv_ import CSVGenerator

    train_generator = CSVGenerator('datasets/ic15/train.csv',
                                   'datasets/ic15/classes.csv',
                                   detect_text=True,
                                   batch_size=1,
                                   phi=5,
                                   shuffle_groups=False)
    misc_effect = MiscEffect()
    for i in range(train_generator.size()):
        image = train_generator.load_image(i)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotations = train_generator.load_annotations(i)
        boxes = annotations['bboxes'].astype(np.int32)
        quadrangles = annotations['quadrangles'].astype(np.int32)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
        cv2.drawContours(image, quadrangles, -1, (0, 255, 255), 1)
        src_image = image.copy()
        # cv2.namedWindow('src_image', cv2.WINDOW_NORMAL)
        cv2.imshow('src_image', src_image)
        # image, annotations = misc_effect(image, annotations)
        image, annotations = multi_scale(image, annotations, prob=1.)
        image = image.copy()
        boxes = annotations['bboxes'].astype(np.int32)
        quadrangles = annotations['quadrangles'].astype(np.int32)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        cv2.drawContours(image, quadrangles, -1, (255, 255, 0), 1)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        cv2.waitKey(0)
