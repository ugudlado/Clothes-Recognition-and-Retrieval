# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:28:33 2019

@author: Wei-Hsiang, Shen
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

from cloth_detection import Detect_Clothes_and_Crop
from utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path


model = Load_DeepFashion2_Yolov3()

img_path = os.path.abspath('./images/test')


onlyfiles = [os.path.join(img_path, f) for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

for file_path in onlyfiles:
    image_name = os.path.basename(file_path)
    # Read image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = Read_Img_2_Tensor(file_path)

    # Clothes detection and crop the image
    images_by_label = Detect_Clothes_and_Crop(img_tensor, model)


    # Transform the image to gray_scale
    # cloth_img = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)

    # # Pretrained classifer parameters
    # PEAK_COUNT_THRESHOLD = 0.02
    # PEAK_VALUE_THRESHOLD = 0.01

    # # Horizontal bins
    # horizontal_bin = np.mean(cloth_img, axis=1)
    # horizontal_bin_diff = horizontal_bin[1:] - horizontal_bin[0:-1]
    # peak_count = len(horizontal_bin_diff[horizontal_bin_diff>PEAK_VALUE_THRESHOLD])/len(horizontal_bin_diff)
    # if peak_count >= PEAK_COUNT_THRESHOLD:
    #     print("Class 1 (clothes wtih stripes)")
    # else:
    #     print("Class 0 (clothes without stripes)")


    plt.imshow(img)
    plt.title('Input image')
    plt.show()

    for label in images_by_label.keys():
        base_dir = f'{img_path}/cropped/{label}'
        Path(base_dir).mkdir(parents=True, exist_ok=True)

    for label,cropped_images in images_by_label.items():
        for cropped_image in cropped_images:
            plt.imshow(cropped_image)
            plt.title(f'Cloth detection and crop - {label}')
            plt.show()
            
            base_dir = f'{img_path}/cropped/{label}'
            Save_Image(cropped_image, f'{base_dir}/cropped_{image_name}')