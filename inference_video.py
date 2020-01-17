from model import efficientdet
import cv2
import os
import numpy as np
from timeit import default_timer as timer
from utils import preprocess_image
from utils.anchors import anchors_for_shape
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

phi = 3
weighted_bifpn = False
model_path = 'checkpoints/2020-01-13/pascal_49_0.8465_0.8430.h5'
image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
image_size = image_sizes[phi] # phi
classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
classes = [
    'hat',
    'nohat',
    'safetybelt',
    'nosafetybelt'
   # 'glove' : 4,
   # 'noglove' : 5,
   # 'boots' : 6,
   # 'noboots' : 7,
   # 'person' : 8
]
num_classes = len(classes)
score_threshold = 0.45
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]

model, prediction_model = efficientdet(phi=phi,
                                       weighted_bifpn=weighted_bifpn,
                                       num_classes=num_classes,
                                       score_threshold=score_threshold)
prediction_model.load_weights(model_path, by_name=True)


def draw(lab,boxes,scores,img,txt,isDetect,isDebug =False):
    """
    用于画框
    :param yolo:
    :param lab:
    :param boxes:
    :param scores:
    :param img:
    :param txt:
    :param isDetect:
    :param isDebug:是否处于调试模式
    :return:
    """
    scores = scores.tolist()
    for i, newbox in enumerate(boxes):
        color = colors[lab[i]] # getColor(lab[i])
        x, y, w, h = 0,0,0,0
        if isDetect:
            x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2] - newbox[0]), int(newbox[3] - newbox[1])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        else:
            x, y, w, h = int(newbox[0]), int(newbox[1]), int(newbox[2]), int(newbox[3])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)

        if isDebug == True:
            text =  ' {} {}: {:.3f}'.format(txt, classes[lab[i]], scores[i])
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 3)
            cv2.rectangle(img, (x, y - text_h - baseline), (x + text_w, y), color, -1)
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img

def detect_image_value(image):
    image = image[:, :, ::-1]
    h, w = image.shape[:2]

    # resize the image into input size
    image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
    # add batch dimension
    inputs = np.expand_dims(image, axis=0)
    anchors = anchors_for_shape((image_size, image_size))
    # run network
    start = timer()
    boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                               np.expand_dims(anchors, axis=0)])
    print(timer() - start)
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
    real_boxes = []
    real_classes = []
    real_scores = []

    for box, score, label in zip(boxes, scores, labels):
        xmin = int(round(box[0]))
        ymin = int(round(box[1]))
        xmax = int(round(box[2]))
        ymax = int(round(box[3]))
        score = '{:.4f}'.format(score)
        class_id = int(label)
        #color = colors[class_id]
        #class_name = classes[class_id]
        #real_label = '-'.join([class_name, score])

        real_boxes.append((ymin, xmin, ymax, xmax))
        real_classes.append(class_id)
        real_scores.append(score)


    return boxes, scores, labels


def detect_camera(video_index=None, videoPath=None, loop=None, output_path="",isDebug=False):
    if (videoPath == None) or (videoPath == '0'):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("The camera is not opening!")
            return
    else:
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Invalid address entered!")
            return
    if loop == None:
        loop = 5
    if loop < 1:
        print("Please enter an integer greater than one.loop:%f", loop)
    video_FourCC = cv2.VideoWriter_fourcc('m', 'p', '4',
                                          'v')  ##int(cap.get(cv2.CAP_PROP_FOURCC))#c mp4v vp09 avc1 hvc1 #
    # video_FourCC    = 0X21
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # drop = video_fps / 15 #保持15帧每秒的最大处理速度，否则丢弃一些帧
    print('the video video_FourCC = {},video_fps= {},  video_size = {}  output = {}'.format(video_FourCC, video_fps,
                                                                                            video_size, output_path))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        pic_path = output_path + '/pic'
        #anno_path = output_path + '/xml'
        out = cv2.VideoWriter(pic_path, video_FourCC, video_fps, video_size)
        # 目标的标签
    frame = 0  # 帧数，循环控制参数
    j = 1
    # start1_time = timer()
    global loc
    global scores
    global lab
    captime = 0
    detecttime = 0
    drawtime = 0

    while True:

        pro_time = timer()
        return_value, img = cap.read()
        captime = captime + (timer() - pro_time)
        if img is None:
            break

        if frame % loop != 0:
            frame = frame + 1
            frame = frame % loop
            continue

        else:
            #img_array = Image.fromarray(img)

            # generate pic file
            if isOutput:
                img_name = '20200106_office_Camera0{}_pic{}.jpg'.format(video_index, j)
                cv2.imwrite(pic_path + '/' + img_name, img)
                j += 1
                # out.write(img)

            # if isDebug == True:
            #     print('start  detect ')
            d_time = timer()
            loc, scores, lab = detect_image_value(img)

            detecttime = detecttime + (timer() - d_time)
            dr_time = timer()
            draw(lab, loc, scores, img, '', True, isDebug=isDebug)
            drawtime = drawtime + (timer() - dr_time)
            result = img

            post_time = timer()
            exec_time = post_time - pro_time
            curr_fps = 1 / exec_time
            fps = '{}: {:.3f}'.format('FPS', curr_fps)


            (fps_w, fps_h), baseline = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (2, 20 - fps_h - baseline), (2 + fps_w, 18), color=(0, 0, 0), thickness=-1)

            cv2.putText(img, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 255), thickness=2)
            if isOutput:
                out.write(result)
            if isDebug == True:
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)

                cv2.imshow("result", result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # quit on ESC button
                if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                    break

        frame = frame + 1
        frame = frame % loop
        # start = True

    cap.release()
    if isOutput:
        out.release()

if __name__ == '__main__':
    detect_camera(loop=1, videoPath='0', isDebug=True)
