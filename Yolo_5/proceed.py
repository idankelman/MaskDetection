
import os
import torch


# print(torch.cuda.is_available())
print(torch.__version__)
filename=input('please enter file name that you want to detect')
os.system('python detect.py --weights best.pt --img 416 --conf 0.4 --source {}'.format(filename))
#os.system('python detect.py --weights best.pt --img 416 --conf 0.4 --source 0')

