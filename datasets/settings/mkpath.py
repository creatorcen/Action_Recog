import csv
import os


file_path = 'train_set_detail.csv'
csv_file = csv.reader(open(file_path))
base_dir = './dataset/frame_and_flow/'

final_detail = []

datatype = 'Frame'
for item in csv_file:
    singl_dir = os.path.join(base_dir, datatype, item[0])
    frame_num = int(item[1])
    label = int(item[2])
    final_detail.append([singl_dir, frame_num, label])
    print([singl_dir, frame_num, label])
print(len(final_detail))
