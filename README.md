# EfficientDet
This is an implementation of [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf) for object detection on Keras and Tensorflow. The project is based on [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
and the [qubvel/efficientnet](https://github.com/qubvel/efficientnet). 
The pretrained EfficientNet weights files are downloaded from [Callidior/keras-applications/releases](https://github.com/Callidior/keras-applications/releases)

Thanks for their hard work.
This project is released under the Apache License. Please take their licenses into consideration too when use this project.

## Train
### build dataset (Pascal VOC, other types please refer to [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet))
* Download VOC2007 and VOC2012, copy all image files from VOC2007 to VOC2012.
* Append VOC2007 train.txt to VOC2012 trainval.txt.
* Overwrite VOC2012 val.txt by VOC2007 val.txt.
### train
* STEP1: `python3 train.py --snapshot imagenet --phi {0, 1, 2, 3, 4, 5, 6} --freeze-backbone --gpu 0 --random-transform --compute-val-loss --batch-size 32 --steps 1000 pascal datasets/VOC2012` to start training. The init lr is 1e-4.
* STEP2: `python3 train.py --snapshot xxx.h5 --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --random-transform --compute-val-loss --batch-size 32 --steps 1000 pascal datasets/VOC2012` to start training when val mAP can not increase during STEP1. The init lr is 1e-5 and decays to 1e-6 when loss stops dropping down.
## Evaluate
* `python3 eval/common.py` to evaluate by specifying model path there.

## Detect
A script is avalable to perform detections with a (self) trained model. It could be used as follows:
`python3 detector.py --model model.h5 --phi=0 --threshold=0.5 --image image.jpg`
