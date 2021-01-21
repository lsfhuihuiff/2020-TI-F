# -*- coding:utf-8 -*-
import cv2


import numpy as np

from keras.models import model_from_json
from anchor_generator import generate_anchors
from anchor_decode import decode_bbox
from nms import single_class_non_max_suppression
import RPi.GPIO as GPIO
model = model_from_json(open('models/face_mask_detection.json').read())
model.load_weights('models/face_mask_detection.hdf5')

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)


# BOARD编号方式，基于插座引脚编号
GPIO.setmode(GPIO.BOARD)
# 输出模式
GPIO.setup(11, GPIO.OUT)  #蜂鸣器
GPIO.setup(13, GPIO.OUT)   # 红灯
GPIO.setup(15, GPIO.OUT)   # 蓝灯
# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)

    result = model.predict(image_exp)

    y_bboxes_output = result[0]
    y_cls_output = result[1]

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                    bbox_max_scores,
                                                    conf_thresh=conf_thresh,
                                                    iou_thresh=iou_thresh,
                                                    )

    for idx in keep_idxs:
        class_id = bbox_max_score_classes[idx]
        if class_id == 0:
            print('戴口罩了')
            GPIO.output(11, GPIO.LOW)
            GPIO.output(13, GPIO.LOW)
            GPIO.output(15, GPIO.HIGH)
        else:
            print('没戴口罩')
            GPIO.output(11, GPIO.HIGH)
            GPIO.output(13, GPIO.HIGH)
            GPIO.output(15, GPIO.LOW)

    return output_info

def run_on_video(video_path, output_video_name, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
        return
    status = True
    print('detection is begining')
    img_id = 0
    while status:
        if img_id%15 == 0:
            print(img_id)
            status, img_raw = cap.read()
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            if (status):
                inference(img_raw,
                                 conf_thresh,
                                 iou_thresh=0.5,
                                 target_shape=(260, 260),
                                 draw_result=True,
                                 show_result=False)

        img_id = img_id +1
if __name__ == "__main__":
    run_on_video(0, '', conf_thresh=0.5)
