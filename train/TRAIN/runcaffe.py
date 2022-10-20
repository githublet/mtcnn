import sys
import os
os.chdir(sys.path[0])
import caffe
caffe.set_mode_gpu()
solver = caffe.SGDSolver('./solver.prototxt')
solver.solve()
