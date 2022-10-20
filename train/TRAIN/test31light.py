# coding=utf-8
import sys
import os
os.chdir(sys.path[0])
import caffe
caffe.set_mode_gpu()
import cv2
import numpy as np
deploy = '31light.prototxt'
caffemodel = '_iter_1300000.caffemodel'
net_31 = caffe.Net(deploy, caffemodel, caffe.TEST)

img_path = "2021 2201641356120.jpg"
img = cv2.imread(img_path)
dim = (31,31)
img = cv2.resize(img, dim,interpolation=cv2.INTER_AREA)
# cv2.imshow("img",img)
# cv2.waitKey(10000)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

caffe_img = (img.copy() - 127.5) / 128
origin_h, origin_w, ch = caffe_img.shape
#print origin_h, origin_w, ch

scale_img = np.swapaxes(caffe_img, 0, 2)
origin_h, origin_w, ch = scale_img.shape
#print origin_h, origin_w, ch

net_31.blobs['data'].reshape(1, 3, 31, 31)
net_31.blobs['data'].data[...] = scale_img

out_ = net_31.forward()
cls_prob = out_['prob1']
print cls_prob[0][1]
