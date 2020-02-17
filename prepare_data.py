import os.path as osp
import cv2
import glob
import os
import json
import logging
import sys
import numpy as np
import random


def rotate(image, vertexes, border_value=(128, 128, 128)):
    rotate_degree = np.random.uniform(low=-45, high=45)
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

    if vertexes is not None and vertexes.shape[0] != 0:
        rotated_vertexes = []
        for vertex in vertexes:
            vertex = np.concatenate([vertex, np.ones((4, 1))], axis=-1)
            rotated_vertex = M.dot(vertex.T)
            rotated_vertexes.append(rotated_vertex.T[:, :2])
        vertexes = np.stack(rotated_vertexes)
    return image, vertexes


def reorder_vertexes(xy_list):
    """
    左上角为起点, 逆时针排序
    Args:
        xy_list:

    Returns:

    """
    reorder_xy_list = np.zeros_like(xy_list)
    # determine the first point with the smallest x,
    # if two has same x, choose that with smallest y,
    ordered_idxes = np.argsort(xy_list, axis=0)
    xmin1_idx = ordered_idxes[0, 0]
    xmin2_idx = ordered_idxes[1, 0]
    if xy_list[xmin1_idx, 0] == xy_list[xmin2_idx, 0]:
        if xy_list[xmin1_idx, 1] <= xy_list[xmin2_idx, 1]:
            reorder_xy_list[0] = xy_list[xmin1_idx]
            first_vertex_idx = xmin1_idx
        else:
            reorder_xy_list[0] = xy_list[xmin2_idx]
            first_vertex_idx = xmin2_idx
    else:
        reorder_xy_list[0] = xy_list[xmin1_idx]
        first_vertex_idx = xmin1_idx
    # connect the first point to others, the third point on the other side of
    # the line with the middle slope
    # others 中存放除去第一个点的其他点在 xy_list 中的下标
    others = list(range(4))
    others.remove(first_vertex_idx)
    # k 用于保存三个斜率值
    k = np.zeros((len(others),))
    for i, vertex_idx in enumerate(others):
        k[i] = (xy_list[vertex_idx, 1] - xy_list[first_vertex_idx, 1]) \
               / (xy_list[vertex_idx, 0] - xy_list[first_vertex_idx, 0] + 0.1e-7)
    # 中间值斜率的坐标
    mid_k_idx = np.argsort(k)[1]
    mid_k = k[mid_k_idx]
    third_vertex_idx = others[mid_k_idx]
    reorder_xy_list[2] = xy_list[third_vertex_idx]
    # determine the second point which on the bigger side of the middle line
    others.remove(third_vertex_idx)
    # y = kx + b
    mid_b = xy_list[first_vertex_idx, 1] - mid_k * xy_list[first_vertex_idx, 0]
    for i, vertex_idx in enumerate(others):
        # delta = y - (k * x + b)
        delta_y = xy_list[vertex_idx, 1] - (mid_k * xy_list[vertex_idx, 0] + mid_b)
        if delta_y > 0:
            # 说明在 13 斜线下方, 作为第四个点
            fourth_vertex_idx = vertex_idx
        else:
            # 说明在 13 斜线上方, 作为第二个点
            second_vertex_idx = vertex_idx
    reorder_xy_list[1] = xy_list[second_vertex_idx]
    reorder_xy_list[3] = xy_list[fourth_vertex_idx]
    # compare slope of 13 and 24, determine the final order
    k13 = mid_k
    k24 = (xy_list[second_vertex_idx, 1] - xy_list[fourth_vertex_idx, 1]) / (
            xy_list[second_vertex_idx, 0] - xy_list[fourth_vertex_idx, 0] + 0.1e-7)
    if k13 < k24:
        # 此时第一个点为左下方的点, 其他三个点按顺时针方向分别存放在 reorder_xy_list 0,1,2,3 的下标位置上
        tmp_x, tmp_y = reorder_xy_list[0, 0], reorder_xy_list[0, 1]
        for i in range(3):
            reorder_xy_list[i] = reorder_xy_list[i + 1]
        reorder_xy_list[3, 0], reorder_xy_list[3, 1] = tmp_x, tmp_y
    return reorder_xy_list


def reorder_vertexes2(vertexes):
    """
    reorder vertexes as the paper shows, (top, right, bottom, left)
    Args:
        vertexes:

    Returns:

    """
    assert vertexes.shape == (4, 2)
    logger.debug(f'vertexes pre order is {vertexes}')
    # 粗暴处理水平或竖直方向有两条线平行的问题, 直接扩展成矩形
    if vertexes[0, 1] == vertexes[1, 1] and vertexes[2, 1] == vertexes[3, 1]:
        vertexes[0, 0] = vertexes[3, 0] = min(vertexes[0, 0], vertexes[3, 0])
        vertexes[1, 0] = vertexes[2, 0] = max(vertexes[1, 0], vertexes[2, 0])
    elif vertexes[0, 0] == vertexes[3, 0] and vertexes[1, 0] == vertexes[2, 0]:
        vertexes[0, 1] = vertexes[1, 1] = min(vertexes[0, 1], vertexes[1, 1])
        vertexes[2, 1] = vertexes[3, 1] = max(vertexes[2, 1], vertexes[3, 1])

    ordered_vertexes = np.zeros_like(vertexes)
    # determine the first point with the smallest y,
    # if two vertexes has same y, choose that with smaller x,
    ordered_idxes = np.argsort(vertexes, axis=0)
    ymin1_idx = ordered_idxes[0, 1]
    ymin2_idx = ordered_idxes[1, 1]
    if vertexes[ymin1_idx, 1] == vertexes[ymin2_idx, 1]:
        if vertexes[ymin1_idx, 0] <= vertexes[ymin2_idx, 0]:
            ordered_vertexes[0] = vertexes[ymin1_idx]
            first_vertex_idx = ymin1_idx
        else:
            ordered_vertexes[0] = vertexes[ymin2_idx]
            first_vertex_idx = ymin2_idx
    else:
        ordered_vertexes[0] = vertexes[ymin1_idx]
        first_vertex_idx = ymin1_idx
    vertexes = np.delete(vertexes, first_vertex_idx, axis=0)
    # determine the second point with the largest x,
    # if two vertexes has same x, choose that with smaller y,
    ordered_idxes = np.argsort(vertexes, axis=0)
    xmax1_idx = ordered_idxes[-1, 0]
    xmax2_idx = ordered_idxes[-2, 0]
    if vertexes[xmax1_idx, 0] == vertexes[xmax2_idx, 0]:
        if vertexes[xmax1_idx, 1] <= vertexes[xmax2_idx, 1]:
            ordered_vertexes[1] = vertexes[xmax1_idx]
            second_vertex_idx = xmax1_idx
        else:
            ordered_vertexes[1] = vertexes[xmax2_idx]
            second_vertex_idx = xmax2_idx
    else:
        ordered_vertexes[1] = vertexes[xmax1_idx]
        second_vertex_idx = xmax1_idx
    vertexes = np.delete(vertexes, second_vertex_idx, axis=0)
    # determine the third point with the largest y,
    # if two vertexes has same y, choose that with larger y,
    if vertexes[0, 1] == vertexes[1, 1]:
        if vertexes[0, 0] <= vertexes[1, 0]:
            ordered_vertexes[2] = vertexes[1]
            ordered_vertexes[3] = vertexes[0]
        else:
            ordered_vertexes[2] = vertexes[0]
            ordered_vertexes[3] = vertexes[1]
    else:
        if vertexes[0, 1] < vertexes[1, 1]:
            ordered_vertexes[2] = vertexes[1]
            ordered_vertexes[3] = vertexes[0]
        else:
            ordered_vertexes[2] = vertexes[0]
            ordered_vertexes[3] = vertexes[1]
    logger.debug(f'vertexes post order is {ordered_vertexes}')
    return ordered_vertexes


def reorder_vertexes3(vertexes):
    """
    reorder vertexes as the paper shows, (top, right, bottom, left)
    强行把四个点拉倒外接矩形对应的边上, 目前采用这种方法
    Args:
        vertexes:

    Returns:

    """
    assert vertexes.shape == (4, 2)
    logger.debug(f'vertexes pre order is {vertexes}')
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
    # 强行把点拉到对应的边上
    ordered_vertexes[0, 1] = ymin
    ordered_vertexes[1, 0] = xmax
    ordered_vertexes[2, 1] = ymax
    ordered_vertexes[3, 0] = xmin
    logger.debug(f'vertexes post order is {ordered_vertexes}')
    return ordered_vertexes


def show_vertex_order_in_labelme():
    dataset_dir = '/home/adam/Pictures/vat/train_quad'
    papery_dataset_dir = osp.join(dataset_dir, 'papery')
    ele_dataset_dir = osp.join(dataset_dir, 'ele')
    for dataset_dir in [papery_dataset_dir, ele_dataset_dir]:
        for image_path in glob.glob(osp.join(dataset_dir, '*.jpg')):
            logger.debug('image_path={}'.format(image_path))
            json_path = image_path[:-4] + '.json'
            if not osp.exists(json_path):
                continue
            # if not image_path.endswith('SKFTE9757TE1901695_7.jpg'):
            #     continue
            labelme_anno = json.load(open(json_path))
            image = cv2.imread(image_path)
            vat_shapes = [shape for shape in labelme_anno['shapes'] if shape['label'] == 'vat']
            if vat_shapes:
                points = vat_shapes[0]['points']
                if len(points) != 4:
                    logger.error(f'{json_path} has vat shapes more than 4 points')
                else:
                    # ordered_points = reorder_vertexes(np.array(points))
                    # ordered_points = reorder_vertexes3(np.array(points))
                    image, vertexes = rotate(image, np.array(points)[None])
                    vertexes = vertexes[0].astype(np.int32)
                    for idx, (point_x, point_y) in enumerate(vertexes):
                        cv2.putText(image, '{}'.format(idx + 1), (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                    (255, 0, 0), 5)
                        cv2.drawContours(image, [vertexes], -1, (255, 0, 0), 2)
                        cv2.drawContours(image, [np.array(points)], -1, (0, 255, 0), 2)

                    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.imshow('image', image)
                    cv2.waitKey(0)
            else:
                logger.warning(f'{json_path} has no vat shapes')


def show_vertex_order_in_via():
    dataset_dir = 'datasets/vat'
    train_dataset_dir = osp.join(dataset_dir, 'train')
    val_dataset_dir = osp.join(dataset_dir, 'val')
    for dataset_dir in [train_dataset_dir, val_dataset_dir]:
        annotations_path = osp.join(dataset_dir, 'via_region_data.json')
        with open(annotations_path) as f:
            annotations = json.load(f)
        for annotation in annotations.values():
            filename = annotation['filename']
            vat_regions = [region for region in annotation['regions'] if region['region_attributes']['name'] == 'vat']
            if vat_regions:
                all_points_x = vat_regions[0]['shape_attributes']['all_points_x']
                all_points_y = vat_regions[0]['shape_attributes']['all_points_y']
                if len(all_points_x) != 4 or len(all_points_y) != 4:
                    logger.error(f'{filename} has regions more than 4 points')
                else:
                    filepath = osp.join(dataset_dir, filename)
                    image = cv2.imread(filepath)
                    for idx, (point_x, point_y) in enumerate(zip(all_points_x, all_points_y)):
                        cv2.putText(image, '{}'.format(idx + 1), (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                    (0, 255, 0), 5)
                    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                    cv2.imshow('image', image)
                    cv2.waitKey(0)

            else:
                logger.warning(f'{filename} has no regions')


def convert_labelme_to_csv():
    dataset_dir = 'datasets/train_quad'
    papery_dataset_dir = osp.join(dataset_dir, 'papery')
    ele_dataset_dir = osp.join(dataset_dir, 'ele')
    papery_image_paths = list(glob.glob(osp.join(papery_dataset_dir, '*.jpg')))
    ele_image_paths = list(glob.glob(osp.join(ele_dataset_dir, '*.jpg')))
    image_paths = papery_image_paths + ele_image_paths
    image_paths = [image_path for image_path in image_paths if osp.exists(image_path[:-4] + '.json')]
    random.shuffle(image_paths)
    train_image_paths = image_paths[:int(len(image_paths) * 0.8)]
    val_image_paths = image_paths[int(len(image_paths) * 0.8):]
    num_train_images = len(train_image_paths)
    num_val_images = len(val_image_paths)
    logger.debug(f'num_train_images={num_train_images}')
    logger.debug(f'num_val_images={num_val_images}')
    train_csv_path = osp.join(dataset_dir, f'train_{num_train_images}_{num_val_images}.csv')
    val_csv_path = osp.join(dataset_dir, f'val_{num_train_images}_{num_val_images}.csv')
    train_csv_file = open(train_csv_path, 'w')
    val_csv_file = open(val_csv_path, 'w')
    for image_paths, csv_file in zip([train_image_paths, val_image_paths], [train_csv_file, val_csv_file]):
        for image_path in image_paths:
            logger.debug('image_path={}'.format(image_path))
            json_path = image_path[:-4] + '.json'
            if not osp.exists(json_path):
                logger.error(f'{json_path} does not exist')
                continue
            labelme_anno = json.load(open(json_path))
            image = cv2.imread(image_path)
            vat_shapes = [shape for shape in labelme_anno['shapes'] if shape['label'] == 'vat']
            if vat_shapes:
                points = vat_shapes[0]['points']
                if len(points) != 4:
                    logger.error(f'{json_path} has vat shapes more than 4 points')
                else:
                    # 判断是否以左上角为起点逆时针标注的
                    ordered_points = reorder_vertexes(np.array(points))
                    if points != ordered_points.tolist():
                        for idx, (point_x, point_y) in enumerate(points):
                            cv2.putText(image, '{}'.format(idx + 1), (point_x, point_y), cv2.FONT_HERSHEY_SIMPLEX, 4,
                                        (0, 255, 0), 5)

                        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                        cv2.imshow('image', image)
                        cv2.waitKey(0)
                    else:
                        # 写入外接矩形
                        points = np.array(points)
                        # min_x, min_y = np.min(points, axis=0)
                        # max_x, max_y = np.max(points, axis=0)
                        # csv_file.write(','.join([image_path, str(min_x), str(min_y), str(max_x), str(max_y), 'vat\n']))
                        # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
                        # 写入四边形
                        points_str = ','.join([str(point_coor) for point_coor in points.reshape(-1).tolist()])
                        csv_file.write(','.join([image_path, points_str, 'vat\n']))
                        # cv2.drawContours(image, [points], -1, (0, 255, 0), 3)
                        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                        # cv2.imshow('image', image)
                        # cv2.waitKey(0)

            else:
                logger.warning(f'{json_path} has no vat shapes')

    train_csv_file.close()
    val_csv_file.close()


if __name__ == '__main__':
    logger = logging.getLogger('prepare_data')
    logger.setLevel(logging.DEBUG)  # default log level
    formatter = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
    sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # show_vertex_order_in_via()
    show_vertex_order_in_labelme()
    # convert_labelme_to_csv()

    # while True:
    #     image = cv2.imread('datasets/train_quad/ele/2_1_011001600111_90113646.jpg')
    #     image, _ = rotate(image, np.zeros((0, 4)))
    #     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)
