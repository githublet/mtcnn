 #coding=utf-8
import os
OK = "F:/TestPIc/xxx/OK.txt"
f1 = open(OK, 'w')
NG = "F:/TestPIc/xxx/NG.txt"
f2 = open(NG, 'w')
Anno = "F:/TestPIc/xxx/Anno.txt"
f3 = open(Anno, 'w')
# 遍历文件夹
def walkFile(file,flag):
    count = 1
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        if flag==0:
            for f in files:
                result = os.path.join(root, f)
                result = result.replace('\\', '/')
                if count == 1:
                    f1.write(result+' 0')
                    f3.write(result+' 0')
                else:
                    f1.write('\n'+result+' 0')
                    f3.write('\n'+result+' 0')
                count +=1
                
        else:
            for f in files:
                result = os.path.join(root, f)
                result = result.replace('\\', '/')
                f2.write('\n'+result+' 1')
                f3.write('\n'+result+' 1')
        # 遍历所有的文件夹
        # for d in dirs:
            # print(os.path.join(root, d))


def main():
    walkFile("F:/TestPIc/xxx/OK",0)
    walkFile("F:/TestPIc/xxx/NG",1)
if __name__ == '__main__':
    main()
