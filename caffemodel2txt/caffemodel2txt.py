# -*- coding: utf-8 -*-

# 引入"caffe"
import sys
#sys.path.append('/home/zt/caffe/build/install/python')
import os
os.chdir(sys.path[0])
import caffe

import numpy as np
 
# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold='nan')
 
# deploy文件
MODEL_FILE = '31light.prototxt'
# 预先训练好的caffe模型
PRETRAIN_FILE = '_iter_1500000.caffemodel'
# 保存参数的文件
params_txt = 'params.txt'
pf = open(params_txt, 'w')
 
# 让caffe以测试模式读取网络参数
net = caffe.Net(MODEL_FILE, PRETRAIN_FILE, caffe.TEST)
 
# 遍历每一层

for param_name in net.params.keys():
    # 该层在prototxt文件中对应"top"的名称
    print(param_name)
    try:
        weight = net.params[param_name][0].data
        shape = weight.shape
        print "Shape: ",shape
        if len(weight.shape) == 4:
            print "Amount, Depth, Height, Width"
            width  = shape[3]
            height = shape[2]
            depth  = shape[1]
            amount = shape[0]
            print(param_name + '_weight:\n')
            for amountCount in range (0, amount):
                if depth == 3:
                    for depthCount in range(depth-1,-1,-1):
                        for widthCount in range (0,width):
                            for heightCount in range (0,height):
                                pf.write('%.8f,\n' % net.params[param_name][0].data[amountCount][depthCount][heightCount][widthCount])
                else:
                    for depthCount in range(0,depth):
                        for widthCount in range (0, width):
                            for heightCount in range (0, height):
                                pf.write('%.8f,\n' % net.params[param_name][0].data[amountCount][depthCount][heightCount][widthCount])
        else:
            weight.shape = (-1, 1)
            
            if (len(weight) == 128*64*3*3):
                C=weight.reshape((128,64,3,3))
                print('73728 w shape',C.shape)
                for amountCount in range (0, 128):
		            for depthCount in range(0,64):
		                for widthCount in range (0,3):
		                    for heightCount in range (0,3):
		                        pf.write('%.8f,\n' % C[amountCount][depthCount][heightCount][widthCount])
            
            else:
                print(param_name + '_weight:\n')
                for w in weight:
                    pf.write('%.8f,\n' % w)
                
    except:
        continue
    # 偏置参数
    try:
        bias = net.params[param_name][1].data
        print (param_name + '_bias:\n')
        bias.shape = (-1, 1)
        for b in bias:
            pf.write('%.8f,\n' % b)
    except:
        continue
pf.close
