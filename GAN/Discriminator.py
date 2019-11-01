import torch
from torch import nn

class Discriminator(nn.Module):
    # 구분자는 이미지를 입력 받아 이미지가 진짜인지 가짜인지 출력한다.
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=256, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid())
    
    # ( batch_size x 1 x 28 x 28 ) 크기의 이미지를 받아
    # 이미지가 진짜일 확률을 0 - 1 사이로 출력한다
    def forward(self, inputs):
        inputs = inputs.view(-1, 28*28)
        return self.main(inputs)


