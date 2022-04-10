# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 16:01:13 2019

@author: Wei-Hsiang, Shen
"""

import time

import cv2
import numpy as np
import tensorflow as tf

from utils_my import (Draw_Bounding_Box, Load_DeepFashion2_Yolov3,
                      Read_Img_2_Tensor)


def Detect_Clothes(img, model_yolov3, eager_execution=True):
    """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
    img = tf.image.resize(img, (416, 416))

    t1 = time.time()
    if eager_execution==True:
        boxes, scores, classes, nums = model_yolov3(img)
        # change eager tensor to numpy array
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()
    else:
        boxes, scores, classes, nums = model_yolov3.predict(img)
    t2 = time.time()
    print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))

    class_names = ['short_sleeve_top', 'long_sleeve_top', 'short_sleeve_outwear', 'long_sleeve_outwear',
                  'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeve_dress',
                  'long_sleeve_dress', 'vest_dress', 'sling_dress']

    # Parse tensor
    list_obj = []
    for i in range(nums[0]):
        obj = {'label':class_names[int(classes[0][i])], 'confidence':scores[0][i]}
        obj['x1'] = boxes[0][i][0]
        obj['y1'] = boxes[0][i][1]
        obj['x2'] = boxes[0][i][2]
        obj['y2'] = boxes[0][i][3]
        list_obj.append(obj)

    return list_obj

def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):
    images_by_label = {}
    list_obj = Detect_Clothes(img_tensor, model)

    img = np.squeeze(img_tensor.numpy())
    img_width = img.shape[1]
    img_height = img.shape[0]

    # crop out one cloth
    for obj in list_obj:
        if obj['label'] == 'short_sleeve_top' and obj['confidence']>threshold:
            img_crop = img[int(obj['y1']*img_height):int(obj['y2']*img_height), int(obj['x1']*img_width):int(obj['x2']*img_width), :]

        if obj['confidence']>threshold:
            cropped_img = img[int(obj['y1']*img_height):int(obj['y2']*img_height), int(obj['x1']*img_width):int(obj['x2']*img_width), :]
            if obj['label'] in images_by_label:
                images_by_label[obj['label']].append(cropped_img)
            else:
                images_by_label[obj['label']] = [cropped_img]
            
    return images_by_label

if __name__ == '__main__':
    img = Read_Img_2_Tensor('./images/test/test7.jpg')
    model = Load_DeepFashion2_Yolov3()
    list_obj = Detect_Clothes(img, model)
    img_with_boxes = Draw_Bounding_Box(img, list_obj)

    cv2.imshow("Clothes detection", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./images/test7_clothes_detected.jpg", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)*255)
