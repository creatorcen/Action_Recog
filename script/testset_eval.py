#!/usr/bin/env python

import os, sys
import collections
import numpy as np
import cv2
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import csv
import json,pickle

sys.path.insert(0, "../")
import models
from VideoSpatialPrediction import VideoSpatialPrediction

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def def_my_result(spat_prediction,layers = 2,topk = 5):
    sort_order = np.argsort(spat_prediction,axis=0)
    input_img_num_fromsingmp4 = np.argsort(spat_prediction,axis=0).shape[1]   #int  250
    pre_result = sort_order[-layers:,:]
    finalpredict = np.reshape(pre_result, (1, input_img_num_fromsingmp4*layers))[0].tolist()
    count = np.zeros(90)
    for i,label in enumerate(finalpredict):
        count[label]=count[label]+1
    final_num = np.sort(count)[-topk:]
    final_label = np.argsort(count)[-topk:]
    return  final_label ,final_num

def write_json(mp4_name,label,score,class_list):
    single_result = []
    single_result.append(mp4_name)
    temp_single_result = []
    for i in range(len(label)):
        temp_single_result.append({"label": class_list[label[-i-1]][:-1], "score": float('%.6f' % score[-i-1])})
    final_result['results'][mp4_name] = temp_single_result

    # with open("./result.json", "w") as file:
    #     json.dump(final_result, file)
    #     file.close()



def main():

    model_path ='/home/thl/Desktop/challeng/checkpoints/675_checkpoint.pth.tar'
    class_name_file = '/home/thl/Desktop/challeng/datasets/settings/class_name.txt'
    class_list = []
    for line in open(class_name_file, "r"):
        class_list.append(line)

    start_frame = 0
    num_categories = 90

    model_start_time = time.time()
    params = torch.load(model_path)

    spatial_net = models.rgb_vgg16(pretrained=False, num_classes=90)
    if torch.cuda.is_available():
        spatial_net = torch.nn.DataParallel(spatial_net)
    spatial_net.load_state_dict(params['state_dict'])
    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))

    val_file_dir = '/home/thl/Desktop/challeng/datasets/settings/test_set.txt'
    val_list = []
    for line in open(val_file_dir, "r"):
        val_list.append(line)

    print("we got %d test videos" % len(val_list))

    line_id = 1

    result_list = []
    for line in val_list:
        clip_path ='/home/thl/Desktop/challeng/datasets/frame_and_flow/test/'+line[:-1]
        spatial_prediction = VideoSpatialPrediction(
                clip_path,
                spatial_net,
                num_categories,
                start_frame)

        final_lab,final_num= def_my_result(spatial_prediction, layers=1)
        # avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        # final_softmax = softmax(final_num/sum(final_num))
        final_softmax =final_num / sum(final_num)
        write_json(line[:-1], final_lab, final_softmax,class_list)
        # result_list.append(avg_spatial_pred_fc8)

        # pred_index = np.argmax(avg_spatial_pred_fc8)

        # print(final_lab,"   ",final_softmax)
        print_score = [float('%.2f' % final_softmax[0]),float('%.2f' % final_softmax[1]),float('%.2f' % final_softmax[2]),
                       float('%.2f' % final_softmax[3]),float('%.2f' % final_softmax[4])]

        print(final_lab,print_score, ' ',line_id ,' / ',len(val_list),'  video ')
        line_id += 1
    print(len(val_list))
    with open("./result.json", "w") as file:
        json.dump(final_result, file)
        file.close()


if __name__ == "__main__":
    final_result = {}
    final_result['results'] = {}
    main()




 
