#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 06:08:10 2019

@author: loktarxiao
"""

import sys

sys.dont_write_bytecode = True
import os


import glob
from tqdm import tqdm
import numpy as np
import cv2
from skimage.measure import label, regionprops
from process_functions import image_read



def preprocess(img_path):
    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    # img = img / 127.5
    # img = img - 1
    return img



def drop_and_fill(mask):
    label_img = label(mask)
    out = np.zeros_like(label_img, dtype=np.float32)

    for region in regionprops(label_img):

        if region.area > 100:
            coord = region.coords
            out[coord[:, 0], coord[:, 1]] = 1.
    for region in regionprops(label(1 - out)):
        if region.area < 100:
            coord = region.coords
            out[coord[:, 0], coord[:, 1]] = 1.

    return out

if __name__ == '__main__':
    IMAGE_SIZE = (1024,1024)

    folder_lst = glob.glob("pre-prediction-5-21/U_Net/U_Net-91/*/")

    save_dir = os.path.join("pre-prediction-full")
    if not os._exists(save_dir): os.makedirs(save_dir, exist_ok=True)


    for folder_path in tqdm(folder_lst):
        print(folder_path)
        # folder_name = folder_path.split("/")[-2]
        folder_name = folder_path.split("/")[-2]
        print(folder_name)
        img_path_lst = glob.glob(folder_path+"*.png")
        coord_lst = [[int(j) for j in i.split("/")[-1].split(".png")[0].split("_")] for i in img_path_lst]
        # coord_lst = [[int(j) for j in i.split("/")[-2].split("_")] for i in img_path_lst]
        all_img = np.stack([preprocess(i) for i in img_path_lst], axis=0)

        # results = tta(all_img, model)[..., 0]
        results = all_img

        output_img = np.zeros((9959, 9958), np.float32)
        count = np.zeros((9959, 9958), np.float32)
        for i, res in zip(coord_lst, results):
            res = cv2.resize(res, (i[1] - i[0], i[3] - i[2]))
            output_img[i[0]:i[1], i[2]:i[3]] += res
            count[i[0]:i[1], i[2]:i[3]] += 1
        result = output_img / count
        # result = np.round(result)
        # result = drop_and_fill(result)
        result = result
        # result = result * 255
        result = result.astype(np.uint8)

        cv2.imwrite(os.path.join(save_dir, folder_name + '.png'), result)



