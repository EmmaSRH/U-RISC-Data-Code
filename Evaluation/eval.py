#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
@author: ruohua
@file: eval.py
@time: 2021/2/22 8:03 PM
"""
import numpy as np
import argparse
from multiprocessing.dummy import Pool as ThreadPool
import os
import cv2
from functools import partial
import glob


def single_eval_boundary(fg_boundary, gt_boundary, bound_pix=0):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        fg_boundary (ndarray): binary boundary prediction image.     Shape: H x W     Value: 0 and 255
        gt_boundary (ndarray): binary annotated boundary image.      Shape: H x W     Value: 0 and 255
        bound_pix: half of the thickness of boundary.
                  A morphology dilation will make the thickness of edge to what we set here
                   this is the radius of disk

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(fg_boundary).shape[2] == 1

    from skimage.morphology import binary_dilation, disk

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F meas
    # ure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F, precision, recall


def reverse_black_and_white(img):
    """
    reverse black and white color
    :param img: binary img
    :return:
    """
    img = 255 - img

    return img


def check_if_white_back_black_edge(pred):
    """
        check if the prediction image is binary
            1. 0 for membrane; 255 for not manbrane
            2. check if binary
    """
    values = np.unique(pred)
    # print(values)

    # check if binary
    if len(values) > 2:
        print("Your prediction result has not been binarized, please prompt them to choose the appropriate threshold for binarization.")
        raise ValueError

    white_pos = np.where(pred == 255)
    # print(len(white_pos[0]))
    white_count = len(white_pos[0])
    black_pos = np.where(pred == 0)
    # print(len(black_pos[0]))
    black_count = len(black_pos[0])
    # print(black_count / white_count)
    rate = black_count / white_count
    if rate < 5:
        print("The results must be submitted with white background and black edge. Please submit after correction.")
        raise ValueError


def single_eval_wrapper(bound_pix, result_matrix, data_list_line):
    id_, pred_path, gt_path = data_list_line

    print("Now evaluating the %s prediction, path = %s" % (id_, pred_path))

    pred_ = cv2.imread(pred_path, 0)
    gt_ = cv2.imread(gt_path, 0)

    pred_ = reverse_black_and_white(pred_)
    gt_ = reverse_black_and_white(gt_)

    check_if_white_back_black_edge(pred_)

    F, precision, recall = single_eval_boundary(pred_, gt_, bound_pix)

    result_matrix[int(id_)] = [F, precision, recall]


def eval_on_whole_dataset(pred_folder_path, gt_folder_path, bound_pix, thread_num):
    pred_list_all =  glob.glob(pred_folder_path + '/*.png')
    print(pred_list_all)

    gt_list_all = []
    for pre in pred_list_all:
        gt_list_all.append(pre.replace(pred_folder_path,gt_folder_path))
    print(gt_list_all)

    id_list = [x for x in range(len(gt_list_all))]

    gt_list = np.asarray(gt_list_all)
    pred_list = np.asarray(pred_list_all)
    id_list = np.asarray(id_list)

    gt_list = np.expand_dims(gt_list, 1)
    pred_list = np.expand_dims(pred_list, 1)
    id_list = np.expand_dims(id_list, 1)

    data_list = np.concatenate([id_list, pred_list, gt_list], axis=1)  # data list for pool

    result_matrix = np.zeros_like(data_list)  # each line contains: F-score, precision, recall

    pool = ThreadPool(thread_num) 

    pool.map(partial(single_eval_wrapper, bound_pix, result_matrix), data_list)

    pool.close()
    pool.join()

    result_matrix = np.asarray(result_matrix, dtype=np.float)
    F_score = result_matrix[:, 0]
    print(F_score)
    mean_F = np.mean(F_score)

    return mean_F

def eval_on_onepic(pred_path, gt_path, bound_pix=0):
    pred_ = cv2.imread(pred_path, 0)
    gt_ = cv2.imread(gt_path, 0)

    pred_ = 255 - pred_
    gt_ = 255 - gt_

    check_if_white_back_black_edge(pred_)
    print("Check format: OK!")

    F, precision, recall = single_eval_boundary(pred_, gt_, bound_pix=0)
    return F

def calc_ap(prec, rec):
    mrec = np.array([0, rec, 1])
    mpre = np.array([0, prec, 0])

    for i in range(mrec.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1]);
        #print mpre[i]

    idx1 = np.where(mrec[1 : ]   != mrec[0 : 2])
    idx2 = [x + 1 for x in idx1]

    ap = sum((mrec.take(idx2) - mrec.take(idx1)) * mpre.take(idx2))
    print("ap = " + str(ap[0]))
    return ap[0]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--evalon1pic', type=str,
                    default='True', help='if evaluate on one image, True or False')
    parser.add_argument('--pre_path', type=str,
                    default='./pre.png', help='prediction path')
    parser.add_argument('--gt_path', type=str,
                    default='./gt.png', help='ground truth path')
    args = parser.parse_args()
    
    
    pre_path = args.pre_path
    gt_path = args.gt_path
    bound_pix = 0  # the thickness of boundary
    thread_num = 4  # number of parallel threads

    print("Evaluate on one image? {}".format(args.evalon1pic))

    if args.evalon1pic == 'True':
        # eval on single image
        print("evaluate {} and {}".format(pre_path,gt_path))
        mean_F = eval_on_onepic(pre_path, gt_path, bound_pix)
        print("The F1-score of prediction is : {}".format(mean_F))

    if args.evalon1pic == 'False':
        # eval on folder
        mean_F = eval_on_whole_dataset(pre_path, gt_path, bound_pix, thread_num)
        print("Mean F_score of all prediction is {}".format(mean_F))


