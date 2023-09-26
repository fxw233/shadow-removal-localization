import glob
import random
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from options import TrainOptions
import matplotlib.pyplot as plt


args = TrainOptions().parse()
transforms_ = [ transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
             transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
ts = [ transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
             transforms.ToTensor()]
ts = transforms.Compose(ts)
transform = transforms.Compose(transforms_)
root = "%s/%s" % (args.data_root,args.dataset_name)
pos1 = 'A'
pos2 = 'B'
files1 = sorted(glob.glob(os.path.join(root, pos1) + '/*.*'))
files2 = sorted(glob.glob(os.path.join(root, pos2) + '/*.*'))
index = 1
imgA = Image.open(files1[index])
imgB = Image.open(files2[index])
img_A = transform(imgA)
print(img_A)
img_B = ts(imgB)
