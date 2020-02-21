"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from generators.common import Generator
import os
import numpy as np
from pycocotools.coco import COCO
import cv2


class CocoGenerator(Generator):
    """
    Generate data from the COCO dataset.
    See https://github.com/cocodataset/cocoapi/tree/master/PythonAPI for more information.
    """

    def __init__(self, data_dir, set_name, **kwargs):
        """
        Initialize a COCO data generator.

        Args
            data_dir: Path to where the COCO dataset is stored.
            set_name: Name of the set to parse.
        """
        self.data_dir = data_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

        super(CocoGenerator, self).__init__(**kwargs)

    def load_classes(self):
        """
        Loads the class to label mapping (and inverse) for COCO.
        """
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        # {'license': 2, 'file_name': '000000259765.jpg', 'coco_url': 'http://images.cocodataset.org/test2017/000000259765.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-21 04:02:31', 'id': 259765}
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32)}

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], np.array([self.coco_label_to_label(a['category_id'])], dtype=np.int32)],
                axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], np.array([[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]], dtype=np.float32)], axis=0)

        return annotations


def show_annotations(generator):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i, group in enumerate(generator.groups):
        images_group, annotations_group = generator.get_augmented_data(group)
        image = images_group[0]
        image[..., 0] *= std[0]
        image[..., 1] *= std[1]
        image[..., 2] *= std[2]
        image[..., 0] += mean[0]
        image[..., 1] += mean[1]
        image[..., 2] += mean[2]
        image = (image * 255.).astype(np.uint8)[:, :, ::-1].copy()
        annotations = annotations_group[0]
        for i in range(annotations['bboxes'].shape[0]):
            bboxes = np.round(annotations['bboxes']).astype(np.int32)[i]
            quadrangles = np.round(annotations['quadrangles']).astype(np.int32)[i]
            alphas = annotations['alphas'][i]
            ratio = annotations['ratios'][i]
            cv2.rectangle(image, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (0, 255, 0), 1)
            cv2.drawContours(image, [quadrangles], -1, (255, 0, 0), 1)
            for i, alpha in enumerate(alphas, 0):
                cv2.putText(image, f'{i}-{alpha:.2f}', (quadrangles[i][0], quadrangles[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0), 1)
            cv2.putText(image, f'{ratio:.2f}', ((bboxes[0] + bboxes[2]) // 2, (bboxes[1] + bboxes[3]) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        cv2.waitKey(0)


def show_targets(generator):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i, group in enumerate(generator.groups):
        inputs, targets, annotations_group = generator.compute_inputs_targets(group, debug=True)

        # image
        image = inputs[0][0]
        image[..., 0] *= std[0]
        image[..., 1] *= std[1]
        image[..., 2] *= std[2]
        image[..., 0] += mean[0]
        image[..., 1] += mean[1]
        image[..., 2] += mean[2]
        image = (image * 255.).astype(np.uint8)[:, :, ::-1].copy()

        # anchor
        batch_regression, batch_class, batch_argmax_overlaps_inds = targets
        regression, classification, argmax_overlaps_inds = batch_regression[0], batch_class[0], batch_argmax_overlaps_inds[0]
        positive_mask = regression[:, -1] == 1
        positive_anchors = generator.anchors[positive_mask].astype(np.int32)
        positive_gt_inds = argmax_overlaps_inds[positive_mask]
        unique_gt_ids, gt_num_anchors = np.unique(positive_gt_inds, return_counts=True)
        bboxes = annotations_group[0]['bboxes'].astype(np.int32)
        for i in range(bboxes.shape[0]):
            if i not in unique_gt_ids or gt_num_anchors[np.where(unique_gt_ids == i)[0][0]] < 5:
                for x1, y1, x2, y2 in positive_anchors[positive_gt_inds == i]:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)

                # gt
                x1, y1, x2, y2 = bboxes[i]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                if generator.detect_quadrangle:
                    quadrangles = annotations_group[0]['quadrangles'].astype(np.int32)
                    cv2.drawContours(image, quadrangles[i], -1, (255, 255, 0), 1)

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    # generator = CSVGenerator('datasets/train_quad/train_800_200.csv',
    #                          'datasets/train_quad/classes.csv',
    #                          batch_size=1, shuffle_groups=False)
    from augmentor.misc import MiscEffect

    generator = CocoGenerator('datasets/coco',
                              'train2017',
                              batch_size=1,
                              phi=0,
                              shuffle_groups=False,
                              )
    # show_annotations(generator)
    # show_targets(generator)
    generator.get_better_ratios_scales(only_base=True)
