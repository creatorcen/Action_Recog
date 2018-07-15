from __future__ import print_function
import os
import glob
import argparse
import cv2
import csv


parser = argparse.ArgumentParser(description="extract frame and optical flows")

parser.add_argument("--out_dir", type=str, default='./frame_and_flow',
                    help='path to store frames and optical flow')

parser.add_argument("--setting_dir",type = str, default = './settings', help = 'path to dataset information document')

parser.add_argument("--class_num",type = str, default = './settings/class_name.txt', help = 'the total class num')

parser.add_argument("--train_set_information",type = str, default = './settings/train_set_detail.txt', help = 'the final trainset detail information')

parser.add_argument("--val_set_information",type = str, default = './settings/val_set_detail.txt', help = 'the final valset detail information')

parser.add_argument("--data_type",type = str, default = 'train', help = 'the type of the data to prepare ,[val, train]')

parser.add_argument("--ext", type=str, default='mp4', choices=['avi','mp4'],
                    help='video file extensions')

args = parser.parse_args()


def singlemp4_creatfolder_video2frame(video_dir,Frame_dir):
    try:
        os.makedirs(Frame_dir+'/Frame')
    except:
        a = input('the file exist clear and restart')
    vc = cv2.VideoCapture(video_dir)
    c = 1
    Frame_dir = Frame_dir+'/Frame'
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        cv2.imwrite(Frame_dir + '/' + 'frame' + '%06d' % c + '.jpg', frame)
        rval, frame = vc.read()
        c = c + 1
    return  c-1



def singlemp4_frame2flow(Frame_dir,Frame_number):
    os.makedirs(Frame_dir + '/Flow_x')
    os.makedirs(Frame_dir + '/Flow_y')
    Base_Frame_dir = Frame_dir + '/Frame/'+ 'frame'

    first_frame_dir = Base_Frame_dir + '%06d' % 1 + '.jpg'
    prev_frame = cv2.imread(first_frame_dir)
    prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    Save_x_dir = Frame_dir + '/Flow_x/' + 'frame'
    Save_y_dir = Frame_dir + '/Flow_y/' + 'frame'

    for frame_num in range(Frame_number-1):

        next_frame_dir = Base_Frame_dir + '%06d' % (frame_num+2) + '.jpg'
        next_frame = cv2.imread(next_frame_dir)
        next_gray_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, next_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        prev_gray_frame = next_gray_frame

        flow_x = flow[..., 0] + 127.5
        flow_y = flow[..., 1] + 127.5

        cv2.imwrite(Save_x_dir+'%06d' % (frame_num+1)+'.jpg', flow_x)
        cv2.imwrite(Save_y_dir+'%06d' % (frame_num+1)+'.jpg', flow_y)




def creat_folder_and_ext_flowframe(datatype):
    if datatype == 'train':
        mp4data_setting_file = args.setting_dir+'/train_set.csv'
        data_num = 20242
    elif datatype == 'val':
        mp4data_setting_file = args.setting_dir+'/val_set.csv'
        data_num = 4050
    else:
        print("Error must be one of the [val,train]")
    data_detail = []
    mp4data_list = csv.reader(open(mp4data_setting_file))
    data_base_dir = './'+datatype+'_set'
    final_data_detail_dir = './settings/'+datatype+'_set_detail.csv'

    num = 1
    for row in mp4data_list:
        print('Process the {} / {},name {}'.format(num,data_num, row[0]))
        singlemp4_dir = data_base_dir + '/'+row[1]+'/'+row[0]+'.'+args.ext
        singlemp4_frame_dir = out_dir+'/'+datatype+'/'+row[0]

        framenumber = singlemp4_creatfolder_video2frame(video_dir = singlemp4_dir, Frame_dir = singlemp4_frame_dir)

        data_detail.append([row[0],framenumber,class_mapping[row[1]],row[1]])

        # singlemp4_frame2flow(Frame_dir = singlemp4_frame_dir,Frame_number = framenumber)

        num = num + 1

    with open(final_data_detail_dir, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_detail)







if __name__ == '__main__':


    out_dir = args.out_dir   # output dir of flow and the frame

    if not os.path.isdir(out_dir):
        print("creating folder: "+out_dir)
        os.makedirs(out_dir)
        os.makedirs(out_dir+'/train')
        os.makedirs(out_dir+'/val')
    else:
        print("The out dataset dir exist! OK Go")

    class_ind = [x for x in enumerate(open(args.class_num))]
    class_mapping = {x[1][:-1] :int(x[0]) for x in class_ind}

    creat_folder_and_ext_flowframe('val')

