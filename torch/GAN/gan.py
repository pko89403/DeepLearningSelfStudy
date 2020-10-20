# 라이브러리 및 데이터 불러오기
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle

from Generator import Generator 
from Discriminator import Discriminator

# 데이터 전처리 방식을 지정한다.
transform = transforms.Compose([
    transforms.ToTensor(), # 데이터를 Pytorch의 Tensor 형식으로 바꾼다.
    transforms.Normalize(mean = (0.5,), std=(0.5,)) # 픽셀 값 ( 0 ~ 1 ) -> ( -1, 1 )
])

# MNIST 데이터 셋을 불러온다. 지정한 폴더에 없을 경우 자동으로 다운로드한다.
mnist = datasets.MNIST(root='data', download=True, transform=transform)

# 데이터를 한번에 batch_size 만큼만 가져오는 dataloader를 만든다.
dataloader = DataLoader(mnist, batch_size=60, shuffle=True)

import os
import imageio

if torch.cuda.is_available():
    use_gpu = True

leave_log = True
if leave_log:
    result_dir = "GAN_generated_images"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

G = Generator()
D = Discriminator()

if use_gpu:
    G.cuda()
    D.cuda()