#!/usr/bin/env python
# coding=utf-8

import os
import cv2
import h5py
import argparse
import numpy as np
import random

sh = 31
sw = 31

ifshow = False
def write_hdf5(file, data, label_class):
  # transform to np array
  data_arr = np.array(data, dtype = np.float32)
  # print data_arr.shape
  # if no swapaxes, transpose to num * channel * width * height ???
  # data_arr = data_arr.transpose(0, 3, 2, 1)
  label_class_arr = np.array(label_class, dtype = np.float32)
  with h5py.File(file, 'w') as f:
    f['data'] = data_arr
    f['label_class'] = label_class_arr

# list_file format:
# image_path | label_class | label_boundingbox(4) | label_landmarks(10)
def convert_dataset_to_hdf5(list_file,  path_save, size_hdf5, tag):
  with open(list_file, 'r') as f:
    annotations = f.readlines()
  num = len(annotations)
  print "%d pics in total" % num
  random.shuffle(annotations)

  data = []
  label_class = []
  count_data = 0
  count_hdf5 = 0
  for line in annotations:
    line_split = line.strip('\n').split('jpg ')
    if len(line_split) != 2:
        print line_split
        continue
    path_full = line_split[0]+'jpg'
    #print path_full
    #cv2.waitKey(10000)
    datum = cv2.imread(path_full)
    classes = float(line_split[1])
    try:
        datum.shape
    except:
        print(path_full)
        continue
    (h,w,c) = datum.shape
    
    if h!=sh or w != sw:
        datum = cv2.resize(datum, (sw, sh),interpolation=cv2.INTER_AREA)
        if classes == 1:
            img_f = cv2.flip(datum,1)
            if ifshow:
                print(path_full+' flip')
                print "classes", classes
                cv2.imshow("img",img_f)
                cv2.waitKey(60000)
            img_f = (img_f - 127.5) * 0.0078125 # [0,255] -> [-1,1]
            img_f = np.swapaxes(img_f, 0, 2)
            data.append(img_f)
            label_class.append(classes)
            count_data = count_data + 1
            if 0 == count_data % size_hdf5:
                path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
                write_hdf5(path_hdf5, data, label_class)
                count_hdf5 = count_hdf5 + 1
                data = []
                label_class = []
                print count_data

    if ifshow:
        print(path_full)
        print "classes", classes
        cv2.imshow("img",datum)
        cv2.waitKey(60000)
        
    # if datum is None:
      # print path_full
      # continue
    #continue
    # normalization
    # BGR to RGB
    # tmp = datum[:, :, 2].copy()
    # datum[:, :, 2] = datum[:, :, 0]
    # datum[:, :, 0] = tmp
    datum = (datum - 127.5) * 0.0078125 # [0,255] -> [-1,1]
    datum = np.swapaxes(datum, 0, 2)
    data.append(datum)
    label_class.append(classes)
    count_data = count_data + 1
    if 0 == count_data % size_hdf5:
      path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
      write_hdf5(path_hdf5, data, label_class)
      count_hdf5 = count_hdf5 + 1
      data = []
      label_class = []
      print count_data
  # handle the rest
  if data:
    path_hdf5 = os.path.join(path_save, tag + str(count_hdf5) + ".h5")
    write_hdf5(path_hdf5, data, label_class)
  print "count_data: %d" % count_data

def main():

  list_file = "F:/TestPIc/xxx/Anno.txt"
  path_save = "D:/TOSHIBA_SSD/JRProject/HDF5/"
  size_hdf5 = 32
  tag = 'train0409_'

  if not os.path.exists(path_save):
    os.makedirs(path_save)
  assert size_hdf5 > 0

  # convert
  convert_dataset_to_hdf5(list_file, path_save, size_hdf5, tag)


if __name__ == '__main__':
  main()
  print "END4CONVERT"
