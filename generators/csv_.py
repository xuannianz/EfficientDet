"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
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
import cv2
import numpy as np
from PIL import Image
from six import raise_from
import csv
import sys
import os.path as osp
from collections import OrderedDict


def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """
    Parse the classes file given by csv_reader.
    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_quadrangle_annotations(csv_reader, classes, detect_text=False):
    """
    Read annotations from the csv_reader.
    Args:
        csv_reader: csv reader of args.annotations_path
        classes: list[str] all the class names read from args.classes_path

    Returns:
        result: dict, dict is like {image_path: [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                     'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4, 'class': class_name}]}

    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader, 1):
        try:
            img_file, x1, y1, x2, y2, x3, y3, x4, y4, class_name = row[:10]
            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, x3, y3, x4, y4, class_name) == ('', '', '', '', '', '', '', '', ''):
                continue

            x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))
            x3 = _parse(x3, int, 'line {}: malformed x3: {{}}'.format(line))
            y3 = _parse(y3, int, 'line {}: malformed y3: {{}}'.format(line))
            x4 = _parse(x4, int, 'line {}: malformed x4: {{}}'.format(line))
            y4 = _parse(y4, int, 'line {}: malformed y4: {{}}'.format(line))

            # check if the current class name is correctly present
            if detect_text:
                if class_name == '###':
                    continue
                else:
                    class_name = 'text'

            if class_name not in classes:
                raise ValueError(f'line {line}: unknown class name: \'{class_name}\' (classes: {classes})')

            result[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                     'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4, 'class': class_name})
        except ValueError:
            raise_from(ValueError(
                f'line {line}: format should be \'img_file,x1,y1,x2,y2,x3,y3,x4,y4,class_name\' or \'img_file,,,,,\''),
                None)

    return result


def _read_annotations(csv_reader, classes):
    """
    Read annotations from the csv_reader.
    Args:
        csv_reader: csv reader of args.annotations_path
        classes: list[str] all the class names read from args.classes_path

    Returns:
        result: dict, dict is like {image_path: [{'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name}]}

    """
    result = OrderedDict()
    for line, row in enumerate(csv_reader, 1):
        try:
            img_file, x1, y1, x2, y2, class_name = row[:10]
            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            if class_name not in classes:
                raise ValueError(f'line {line}: unknown class name: \'{class_name}\' (classes: {classes})')

            result[img_file].append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name})
        except ValueError:
            raise_from(ValueError(
                f'line {line}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''),
                None)

    return result


def _open_for_csv(path):
    """
    Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb', for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class CSVGenerator(Generator):
    """
    Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
            self,
            csv_data_file,
            csv_class_file,
            base_dir=None,
            detect_quadrangle=False,
            detect_text=False,
            **kwargs
    ):
        """
        Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            detect_text: if do text detection
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        self.detect_quadrangle = detect_quadrangle
        self.detect_text = detect_text

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            if osp.exists(csv_data_file):
                self.base_dir = ''
            else:
                self.base_dir = osp.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                # class_name --> class_id
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        # class_id --> class_name
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, x3, y3, x4, y4, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                # {'img_path1':[{'x1':xx,'y1':xx,'x2':xx,'y2':xx,'x3':xx,'y3':xx,'x4':xx,'y4':xx, 'class':xx}...],...}
                if self.detect_quadrangle:
                    self.image_data = _read_quadrangle_annotations(csv.reader(file, delimiter=','), self.classes,
                                                                   self.detect_text)
                else:
                    self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(detect_text=detect_text, detect_quadrangle=detect_quadrangle, **kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """
        Returns the image path for image_index.
        """
        return osp.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        image = cv2.imread(self.image_path(image_index))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        path = self.image_names[image_index]
        annotations = {'labels': np.empty((0,), dtype=np.int32),
                       'bboxes': np.empty((0, 4), dtype=np.float32),
                       'quadrangles': np.empty((0, 4, 2), dtype=np.float32),
                       }

        for idx, annot in enumerate(self.image_data[path]):
            annotations['labels'] = np.concatenate((annotations['labels'], [self.name_to_label(annot['class'])]))
            if self.detect_quadrangle:
                quadrangle = np.array([[float(annot['x1']), float(annot['y1'])],
                                       [float(annot['x2']), float(annot['y2'])],
                                       [float(annot['x3']), float(annot['y3'])],
                                       [float(annot['x4']), float(annot['y4'])]])
                ordered_quadrangle = self.reorder_vertexes(quadrangle)
                annotations['quadrangles'] = np.concatenate((annotations['quadrangles'], ordered_quadrangle[None]))
                annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                    float(min(annot['x1'], annot['x2'], annot['x3'], annot['x4'])),
                    float(min(annot['y1'], annot['y2'], annot['y3'], annot['y4'])),
                    float(max(annot['x1'], annot['x2'], annot['x3'], annot['x4'])),
                    float(max(annot['y1'], annot['y2'], annot['y3'], annot['y4'])),
                ]]))
            else:
                annotations['bboxes'] = np.concatenate((annotations['bboxes'], [[
                    float(annot['x1']),
                    float(annot['y1']),
                    float(annot['x2']),
                    float(annot['y2']),
                ]]))
        return annotations

    def reorder_vertexes(self, vertexes):
        """
        reorder vertexes as the paper shows, (top, right, bottom, left)
        Args:
            vertexes:

        Returns:

        """
        assert vertexes.shape == (4, 2)
        xmin, ymin = np.min(vertexes, axis=0)
        xmax, ymax = np.max(vertexes, axis=0)

        # determine the first point with the smallest y,
        # if two vertexes has same y, choose that with smaller x,
        ordered_idxes = np.argsort(vertexes, axis=0)
        ymin1_idx = ordered_idxes[0, 1]
        ymin2_idx = ordered_idxes[1, 1]
        if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
            if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
                first_vertex_idx = ymin1_idx
            else:
                first_vertex_idx = ymin2_idx
        else:
            first_vertex_idx = ymin1_idx
        ordered_idxes = [(first_vertex_idx + i) % 4 for i in range(4)]
        ordered_vertexes = vertexes[ordered_idxes]
        # drag the point to the corresponding edge
        ordered_vertexes[0, 1] = ymin
        ordered_vertexes[1, 0] = xmax
        ordered_vertexes[2, 1] = ymax
        ordered_vertexes[3, 0] = xmin
        return ordered_vertexes
