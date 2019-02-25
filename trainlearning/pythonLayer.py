import sys
sys.path.append('/home/zt/caffe/build/install/python/')
print sys.path
import caffe
import cv2
import numpy as np
import random

imdb_exit = False

def view_bar(num, total):
  rate = float(num) / total
  rate_num = int(rate * 100)
  r = '\r[%s%s]%d%%' % ("#" * rate_num, " " * (100 - rate_num), rate_num, )
  sys.stdout.write(r)
  sys.stdout.flush()

################################################################################
#########################Data Layer By Python###################################
################################################################################
class Data_Layer_train(caffe.Layer):
  def setup(self, bottom, top):
    self.batch_size = 4
    net_side = 12
    cls_list = './4.txt'
    roi_list = './5.txt'
    pts_list = ''
    cls_root = './'
    roi_root = './'
    pts_root = ''
    self.batch_loader = BatchLoader(cls_list, roi_list, pts_list, net_side, cls_root, roi_root, pts_root)
    top[0].reshape(self.batch_size, 3, net_side, net_side)
    top[1].reshape(self.batch_size, 1)
    top[2].reshape(self.batch_size, 4)
    # top[3].reshape(self.batch_size, 10)

  def reshape(self, bottom, top):
    pass

  def forward(self, bottom, top):
    loss_task = random.randint(0, 1)
    # loss_task = 0
    for itt in range(self.batch_size):
      im, label, roi, pts = self.batch_loader.load_next_image(loss_task)
      top[0].data[itt, ...] = im
      top[1].data[itt, ...] = label
      top[2].data[itt, ...] = roi
      # top[3].data[itt, ...] = pts

  def backward(self, top, propagate_down, bottom):
    pass


class BatchLoader(object):
  def __init__(self, cls_list, roi_list, pts_list, net_side, cls_root, roi_root, pts_root):
    self.mean = 127.5
    self.im_shape = net_side
    self.cls_root = cls_root
    self.roi_root = roi_root
    self.pts_root = pts_root
    self.roi_list = []
    self.cls_list = []
    self.pts_list = []
    print
    "Start Reading Classify Data into Memory..."
    if imdb_exit:
      fid = open('12/cls.imdb', 'r')
      self.cls_list = pickle.load(fid)
      fid.close()
    else:
      fid = open(cls_list, 'r')
      lines = fid.readlines()
      fid.close()
      cur_ = 0
      sum_ = len(lines)
      for line in lines:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = self.cls_root + words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != self.im_shape or w != self.im_shape:
          im = cv2.resize(im, (int(self.im_shape), int(self.im_shape)))
        im= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.swapaxes(im, 0, 2)
        im = (im - self.mean) * 0.0078125
        label = int(words[1])
        roi = [-1, -1, -1, -1]
        pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.cls_list.append([im, label, roi, pts])
    random.shuffle(self.cls_list)
    self.cls_cur = 0
    print
    "\n", str(len(self.cls_list)), " Classify Data have been read into Memory..."

    print
    "Start Reading Regression Data into Memory..."
    if imdb_exit:
      fid = open('12/roi.imdb', 'r')
      self.roi_list = pickle.load(fid)
      fid.close()
    else:
      fid = open(roi_list, 'r')
      lines = fid.readlines()
      fid.close()
      cur_ = 0
      sum_ = len(lines)
      for line in lines:
        view_bar(cur_, sum_)
        cur_ += 1
        words = line.split()
        image_file_name = self.roi_root + words[0] + '.jpg'
        im = cv2.imread(image_file_name)
        h, w, ch = im.shape
        if h != self.im_shape or w != self.im_shape:
          im = cv2.resize(im, (int(self.im_shape), int(self.im_shape)))
        im= cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.swapaxes(im, 0, 2)
        im = (im - self.mean) * 0.0078125
        label = int(words[1])
        roi = [float(words[2]), float(words[3]), float(words[4]), float(words[5])]
        pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        self.roi_list.append([im, label, roi, pts])
    random.shuffle(self.roi_list)
    self.roi_cur = 0
    print
    "\n", str(len(self.roi_list)), " Regression Data have been read into Memory..."


  def load_next_image(self, loss_task):
    if loss_task == 0:
      if self.cls_cur == len(self.cls_list):
        self.cls_cur = 0
        random.shuffle(self.cls_list)
      cur_data = self.cls_list[self.cls_cur]  # Get the image index
      im = cur_data[0]
      label = cur_data[1]
      roi = [-1, -1, -1, -1]
      pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
      if random.choice([0, 1]) == 1:
        im = cv2.flip(im, random.choice([-1, 0, 1]))
      self.cls_cur += 1
      return im, label, roi, pts

    if loss_task == 1:
      if self.roi_cur == len(self.roi_list):
        self.roi_cur = 0
        random.shuffle(self.roi_list)
      cur_data = self.roi_list[self.roi_cur]  # Get the image index
      im = cur_data[0]
      label = -1
      roi = cur_data[2]
      pts = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
      self.roi_cur += 1
      return im, label, roi, pts

    if loss_task == 2:
      if self.pts_cur == len(self.pts_list):
        self.pts_cur = 0
        random.shuffle(self.pts_list)
      cur_data = self.pts_list[self.pts_cur]  # Get the image index
      im = cur_data[0]
      label = -1
      roi = [-1, -1, -1, -1]
      pts = cur_data[3]
      self.pts_cur += 1
      return im, label, roi, pts


################################################################################
#########################ROI Loss Layer By Python###############################
################################################################################
class regression_Layer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")
    
  def reshape(self, bottom, top):
    if bottom[0].count != bottom[1].count:
      raise Exception("Input predict and groundTruth should have same dimension")
    roi = bottom[1].data
    print 'len(roi[:,0]) ++++>>> ', len(roi[:,0]) 
    self.valid_index = np.where(roi[:, 0] != -1)[0]
    self.N = len(self.valid_index)
    self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
    top[0].reshape(1)

  def forward(self, bottom, top):
    self.diff[...] = 0
    top[0].data[...] = 0
    if self.N != 0:
      self.diff[...] = bottom[0].data - np.array(bottom[1].data).reshape(bottom[0].data.shape)
      top[0].data[...] = np.sum(self.diff ** 2) / bottom[0].num / 2.

  def backward(self, top, propagate_down, bottom):
    
    for i in range(2):
      if not propagate_down[i] or self.N == 0:
        continue
      if i == 0:
        sign = 1
      else:
        sign = -1
      bottom[i].diff[...] = sign * self.diff / bottom[i].num
      print "****ROI bottom[i].diff**\n",bottom[i].diff


################################################################################
#############################SendData Layer By Python###########################
################################################################################
class cls_Layer_fc(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
      raise Exception("Need 2 Inputs")

  def reshape(self, bottom, top):
    top[0].reshape(len(bottom[0].data), 2, 1, 1)
    top[1].reshape(len(bottom[1].data),1)
  def forward(self, bottom, top):
    top[0].data[...][...] = 0
    top[1].data[...][...] = 0
    top[0].data[...] = bottom[0].data[...]
    #print "\n\n*****forward top[0] *****\n\n",top[0].data
    top[1].data[...] = bottom[1].data[...]
    #print "\n\n*****forward top[1] *****\n\n",top[1].data

  def backward(self, top, propagate_down, bottom):
    bottom[0].diff[...] = 0
    bottom[0].diff[...] = top[0].diff[...]
    #print "\n\n*****backward top[0] *****\n\n",top[0].diff

class MyLayer(caffe.Layer):
  def setup(self, bottom, top):
    pass

  def reshape(self, bottom, top):
    top[0].reshape(*bottom[0].data.shape)

  def forward(self, bottom, top):
    top[0].data[...] = bottom[0].data[:]
    print "*****mylayer top[0].data*****\n" ,top[0].data
  def backward(self, top, propagate_down, bottom):
    for i in range(len(propagate_down)):
        if not propagate_down[i]:
            continue
        bottom[i].diff[...] = top[i].diff[:]
        print "*******mylayer  bottom[i].diff*****\n" , bottom[i].diff

