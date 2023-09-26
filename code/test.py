import argparse
import os
import numpy as np
import time
import datetime
import sys
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from models import Create_nets
from datasets import Get_dataloader
from options import TrainOptions
from optimizer import *
from utils import sample_images , LambdaLR

def thresold(output_s , t):
    output_s[output_s > t] = 1.0
    output_s[output_s <= t] = 0.0
    return output_s
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
args = TrainOptions().parse()
# Calculate output of image discriminator (PatchGAN)
D_out_size = 256//(2**args.n_D_layers) - 2

patch = (1, D_out_size, D_out_size)
criterion_GAN, criterion_pixelwise = Get_loss_func(args)
# Initialize generator and discriminator
print( "#################################################################################")
generator, discriminator = Create_nets(args)
generator.eval()
path1 = 'result/generator2_199.pth'
train_dataloader,test_dataloader,_ = Get_dataloader(args)
generator.load_state_dict(torch.load(path1))

for i, batch in enumerate(test_dataloader):
    real_A = Variable(batch['A'].type(torch.FloatTensor).cuda())  
    filename = batch['B']  
    fake_B = generator(real_A)
    output_s = torch.sigmoid(fake_B)
    output_s = output_s.data.cpu().squeeze(0)

    image_w , image_h = 480, 640
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_w, image_h))
    ])
    output_s = transform(output_s)
    
    # save saliency maps
    save_test_path = './pred/ISTD_new/'
    if not os.path.exists(save_test_path):
        os.makedirs(save_test_path)
    output_s.save(os.path.join(save_test_path, filename[0]))