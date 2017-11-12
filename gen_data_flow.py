import numpy as np  
import cv2  
from collections import deque
from PIL import Image
import tensorflow as tf
from config import *
import sys
import struct
import feature_fetcher
import os
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=str)
parser.add_argument('--start-file-num', type=int, default=0)
args = parser.parse_args()

#data_names = ["test", "train"]
data_names = ["train"]
data_path = "data6/"
before_ch = 7
after_ch = 7
tot_ch = before_ch + after_ch + 1
for dn in data_names:
    list_f = open('data_video/' + dn + '_list_', 'r')
    temp = list_f.read()
    video_list = temp.split('\n')

    record_num = 0
    file_num = args.start_file_num
    file_list = str(file_num) + ".tfrecords"
    writer = tf.python_io.TFRecordWriter(data_path + dn + "/" + str(file_num) + ".tfrecords")

    start = 0
    if args.start is not None:
        for idx, video_name in enumerate(video_list):
            if video_name == args.start:
                start = idx
                break
    for idx in range(start, len(video_list)):
        video_name = video_list[idx]
        if (video_name == ""):
            break
        print('processing video '+video_name)
        flowfile = open('data_video/flow/' + video_name[:-4] + '.bin', 'rb')
        flowdata = flowfile.read()
        float_cnt = 4
        cnt = 2 * height * width * float_cnt * before_ch

        stable_cap = cv2.VideoCapture('data_video/stable/' + video_name)  
        unstable_cap = cv2.VideoCapture('data_video/unstable/' + video_name)  
        assert(os.path.isfile('data_video/stable/' + video_name))
        assert(os.path.isfile('data_video/unstable/' + video_name))
        unstable_frames = []
        stable_frames = []
        print('data_video/stable/' + video_name)
        for i in range(tot_ch + 1):
            ret, frame = stable_cap.read()
            stable_frames.append(cvt_img2train(frame, crop_rate))
            ret, frame = unstable_cap.read()
            unstable_frames.append(cvt_img2train(frame, 1))
        length = 0
        while(True):
            length += 1
            if (length % 10 == 0):
                print(length)
            #if (length == 30):
            #    break
            stable = stable_frames[0] #[0:before_ch+2]=[0:9], 9
            for i in range(1, before_ch + 2):
                stable = np.concatenate((stable, stable_frames[i]), axis=3)
            unstable = unstable_frames[before_ch] #[before_ch:tot_ch+1]=[7:16], 9
            for i in range(before_ch + 1, tot_ch + 1):
                unstable = np.concatenate((unstable, unstable_frames[i]), axis=3)
            #calc flow_x
            flow = np.zeros((height, width, 2), dtype=np.float32)
            for xx in range(height):
                for yy in range(width):
                    bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
                    cnt += float_cnt
                    flow[xx, yy, 0]=bit + yy
            flow[:, :, 0] = flow[:, :, 0] / width * 2 - 1
            #calc flow_y
            for xx in range(height):
                for yy in range(width):
                    bit = float(struct.unpack('f', flowdata[cnt:cnt+float_cnt])[0])
                    cnt += float_cnt
                    flow[xx, yy, 1]=bit + xx
            flow[:, :, 1] = flow[:, :, 1] / height * 2 - 1

            stable = stable.flatten().tolist()
            unstable = unstable.flatten().tolist()
            flow = flow.flatten().tolist()
            example = tf.train.Example(features=tf.train.Features(feature={
                "stable": tf.train.Feature(float_list=tf.train.FloatList(value=stable)),
                "unstable": tf.train.Feature(float_list=tf.train.FloatList(value=unstable)),
                "flow": tf.train.Feature(float_list=tf.train.FloatList(value=flow)),
                "feature_matches1": tf.train.Feature(float_list=tf.train.FloatList(value=\
                    feature_fetcher.fetch(video_name, before_ch + length - 1).flatten().tolist())),
                "feature_matches2": tf.train.Feature(float_list=tf.train.FloatList(value=\
                    feature_fetcher.fetch(video_name, before_ch + length).flatten().tolist())),
            }))

            writer.write(example.SerializeToString())
            record_num += 1
            if (record_num == tfrecord_item_num):
                record_num = 0
                file_num += 1
                writer.close()
                file_list += " " + str(file_num) + ".tfrecords"
                writer = tf.python_io.TFRecordWriter(data_path + dn + "/" + str(file_num) + ".tfrecords")

            ret, frame_stable = stable_cap.read()  
            ret, frame_unstable = unstable_cap.read()
            if (not ret):
                break
            stable_frames.append(cvt_img2train(frame_stable, crop_rate))
            unstable_frames.append(cvt_img2train(frame_unstable, 1))
            stable_frames.pop(0)
            unstable_frames.pop(0)
        stable_cap.release()  
        unstable_cap.release()  
    writer.close()
    file_object = open(data_path + dn + "/list.txt", 'w')
    file_object.write(file_list)
    file_object.close( )
