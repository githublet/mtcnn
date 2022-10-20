import sys
#sys.path.append('C:\"Program Files"\Anaconda3\envs\py27')
import caffe
import cv2
import numpy as np
import random
count = 0;
################################################################################
#########################ROI Loss Layer By Python###############################
################################################################################
class normMSE(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")

  def reshape(self, bottom, top):
    if bottom[0].count != bottom[1].count:
      raise Exception("Input predict and groundTruth should have same dimension")
    roi = bottom[1].data
    self.valid_index = np.where(roi[:, 0] != -1)[0]
    self.N = len(self.valid_index)
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = 0
    top[0].data[...] = 0
    if self.N != 0:
      delX = bottom[1].data[self.valid_index,0]-bottom[1].data[self.valid_index,1]
      delY = bottom[1].data[self.valid_index,5]-bottom[1].data[self.valid_index,6]

      self.norm = (1e-6+(delX*delX + delY*delY)**0.5)

      diff = bottom[0].data[self.valid_index] - np.array(bottom[1].data[self.valid_index]).reshape(bottom[0].data[self.valid_index].shape)
      self.diff[self.valid_index] = ((diff.T)/(self.norm)).T
      top[0].data[...] = np.sum(self.diff ** 2) /self.N/ 2.

  def backward(self, top, propagate_down, bottom):
    for i in range(2):
      if not propagate_down[i] or self.N == 0:
        continue
      if i == 0:
        sign = 1
      else:
        sign = -1
      bottom[i].diff[...] = 0
      bottom[i].diff[...]= sign * self.diff / bottom[0].num # normalization: BATCH_SIZE bottom[0].num normalization: VALID self.N

class regression_Layer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")

  def reshape(self, bottom, top):
    if bottom[0].count != bottom[1].count:
      raise Exception("Input predict and groundTruth should have same dimension")
    roi = bottom[1].data
    if len(roi[0, :]) == 4:
      self.loss_weight = 0.5
    else:
      self.loss_weight = 1.5
    self.valid_index = np.where(roi[:, 0] != -1)[0]
    self.N = len(self.valid_index)
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = 0
    top[0].data[...] = 0
    if self.N != 0:
      self.diff[self.valid_index] = bottom[0].data[self.valid_index] - np.array(bottom[1].data[self.valid_index]).reshape(bottom[0].data[self.valid_index].shape)
      top[0].data[...] = np.sum(self.diff[self.valid_index] ** 2) /float(self.N)/ 2.
      
  def backward(self, top, propagate_down, bottom):
    for i in range(2):
      if not propagate_down[i] or self.N == 0:
        continue
      if i == 0:
        sign = 1
      else:
        sign = -1
      bottom[i].diff[...] = 0
      bottom[i].diff[...] = sign * self.diff * self.loss_weight / float(self.N) # normalization: BATCH_SIZE bottom[0].num normalization: VALID self.N

    #online positive
class regression_Layer_op(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 3:
      raise Exception("Need 3 Inputs")

  def reshape(self, bottom, top):
    if bottom[0].count != bottom[1].count:
      raise Exception("Input predict and groundTruth should have same dimension")
    pts = bottom[1].data
    cls = bottom[2].data
    a = np.exp(cls)
    b = np.sum(a,axis=1)+1e-6
    c = np.divide(a[:,1],b)
    valid_cls = np.where(c > 0.5)[0]
    valid_pts = np.where(pts[:, 0] != -1)[0]
    self.loss_weight = 1.0
    self.valid_index = np.intersect1d(valid_cls,valid_pts)
    self.N = len(self.valid_index)
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = 0
    top[0].data[...] = 0
    if self.N != 0:
      self.diff[self.valid_index] = bottom[0].data[self.valid_index] - np.array(bottom[1].data[self.valid_index]).reshape(bottom[0].data[self.valid_index].shape)
      top[0].data[...] = np.sum(self.diff[self.valid_index] ** 2) /float(self.N)/ 2.
      
  def backward(self, top, propagate_down, bottom):
    for i in range(2):
      if not propagate_down[i] or self.N == 0:
        continue
      if i == 0:
        sign = 1
      else:
        sign = -1
      bottom[i].diff[...] = 0
      bottom[i].diff[...] = sign * self.diff * self.loss_weight / float(self.N) # normalization: BATCH_SIZE bottom[0].num normalization: VALID self.N
      
    #online hard just way to implement , may be no use 
class regression_Layer_oh(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")

  def reshape(self, bottom, top):
    if bottom[0].count != bottom[1].count:
      raise Exception("Input predict and groundTruth should have same dimension")
    self.valid_index = np.where(bottom[1].data[:, 0] != -1)[0]
    self.loss_weight = 1.0
    self.N = int(len(self.valid_index) * 0.5)
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    self.tmp = np.zeros_like(bottom[0].data, dtype=np.float32)
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = 0
    top[0].data[...] = 0
    if self.N != 0:
      self.tmp[self.valid_index] = bottom[0].data[self.valid_index] - np.array(bottom[1].data[self.valid_index]).reshape(bottom[0].data[self.valid_index].shape)
      euclidean = np.sum(self.tmp ** 2,axis =1)
      ohemO = np.argsort(-euclidean)[0:self.N]
      self.diff[ohemO] = self.tmp[ohemO]
      top[0].data[...] = np.sum(euclidean[ohemO])/float(self.N)/ 2.
      
  def backward(self, top, propagate_down, bottom):
    for i in range(2):
      if not propagate_down[i] or self.N == 0:
        continue
      if i == 0:
        sign = 1
      else:
        sign = -1
      bottom[i].diff[...] = 0
      bottom[i].diff[...] = sign * self.diff * self.loss_weight / float(self.N) # normalization: BATCH_SIZE bottom[0].num normalization: VALID self.N
################################################################################
#############################SendData Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")

  def reshape(self, bottom, top):
    top[0].reshape(len(bottom[0].data), 2,1,1)
    top[1].reshape(len(bottom[1].data))

  def forward(self, bottom, top):
    top[0].data[...][...] = 0
    top[1].data[...][...] = 0
    top[0].data[...] = bottom[0].data[...]
    top[1].data[...] = bottom[1].data[...]


  def backward(self, top, propagate_down, bottom):
    if propagate_down[0]:
      bottom[0].diff[...] = 0
      bottom[0].diff[...] = top[0].diff[...]

class cls_Layer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")

  def reshape(self, bottom, top):
    top[0].reshape(len(bottom[0].data), 2)
    top[1].reshape(len(bottom[1].data))

  def forward(self, bottom, top):
    top[0].data[...][...] = 0
    top[1].data[...][...] = 0
    top[0].data[...] = bottom[0].data[...]
    top[1].data[...] = bottom[1].data[...]


  def backward(self, top, propagate_down, bottom):
    if propagate_down[0]:
      bottom[0].diff[...] = 0
      bottom[0].diff[...] = top[0].diff[...]
      #print("\nYeap Yeap Yeap!!!\n")
      
      
class cls_Layer_ohem(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")

  def reshape(self, bottom, top):
    top[0].reshape(len(bottom[0].data), 2)
    top[1].reshape(len(bottom[1].data))

  def forward(self, bottom, top):
    global count
    count +=1
    top[0].data[...][...] = 0
    top[1].data[...][...] = -1
    #softmax
    #if count%100000>50000 and count%100000 < 70000:
    if random.randint(1,5)==3:
      a = np.exp(bottom[0].data)
      b = np.sum(a,axis=1)
      a[:,0] = np.divide(a[:,0],b)
      a[:,1] = np.divide(a[:,1],b)
      #ohem
      label = bottom[1].data
      self.valid_index = np.where(label != -1)[0]
      self.count = len(self.valid_index)
      ohemN = int(self.count * 0.7)
      label_ = np.zeros_like(bottom[1].data)
      label_[self.valid_index] = label[self.valid_index]
      ohem = np.ones_like(bottom[1].data.reshape(-1),dtype=np.float32)
      ohem[self.valid_index] = (np.where(label_.reshape(-1),a[:,1].reshape(-1),a[:,0].reshape(-1)))[self.valid_index]
      ohemO = np.argsort(ohem)[0:ohemN]
      top[0].data[...] = bottom[0].data[...]
      top[1].data[ohemO] = bottom[1].data[ohemO]
      if count %50000 == 1:
        print top[0].data[...]
        print top[1].data[...]
    else:
      top[0].data[...] = bottom[0].data[...]
      top[1].data[...] = bottom[1].data[...]

  def backward(self, top, propagate_down, bottom):
    if propagate_down[0]:
      bottom[0].diff[...] = 0
      bottom[0].diff[...] = top[0].diff[...]
      #print("\nYeap Yeap Yeap!!!\n")
