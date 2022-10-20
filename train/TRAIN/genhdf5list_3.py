 #coding=utf-8
import os
Anno = "D:/TOSHIBA_SSD/JRProject/TRAIN/hdf5_list.txt"
f1 = open(Anno, 'w')
# 遍历文件夹

def walkFile(file):
    count = 1
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        for f in files:
            result = os.path.join(root, f)
            result = result.replace('\\', '/')
            if count == 1:
                f1.write(result)
            else:
                f1.write('\n'+result)
            count +=1


def main():
    walkFile("D:/TOSHIBA_SSD/JRProject/HDF5")
if __name__ == '__main__':
    main()
