export PYTHONPATH="/home/zt/mtcnn_train/12net/trainlearning:$PYTHONPATH"
echo $PYTHONPATH
/home/zt/caffe/build/tools/caffe train --solver=./solver.prototxt -weights det1.caffemodel -gpu 0
