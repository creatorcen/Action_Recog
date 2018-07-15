from __future__ import print_function
import os
import glob
import argparse
import cv2
import csv


test_list_dir = './settings/test_set.txt'

data_name_list = []


def singlemp4_creatfolder_video2frame(video_dir, Frame_dir):
    try:
        os.makedirs(Frame_dir + '/Frame')
    except:
        a = input('the file exist clear and restart')
    vc = cv2.VideoCapture(video_dir)
    c = 1
    Frame_dir = Frame_dir + '/Frame'
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        cv2.imwrite(Frame_dir + '/' + 'frame' + '%06d' % c + '.jpg', frame)
        rval, frame = vc.read()
        c = c + 1
    # return c - 1


if __name__ == '__main__':

    for line in open(test_list_dir, "r"):
        data_name_list.append(line)
    # print(type(data_name_list))
    # print(len(data_name_list))
    len_test_dataset = len(data_name_list)
    for i, mp4name in enumerate(data_name_list):
        single_mp4_dir = './test_set/' + mp4name[:-1] + '.mp4'
        Frame_dir = './frame_and_flow/test/' + mp4name[:-1]
        # print(single_mp4_dir)
        singlemp4_creatfolder_video2frame(single_mp4_dir, Frame_dir)
        print('Process ',i+1,'/ ',len_test_dataset,'fime name ',mp4name)
        # print(Frame_dir)
