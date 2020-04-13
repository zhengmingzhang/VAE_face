import torch
import torch.nn as nn
# 定义自编码器的网络结构
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # self.encoder=nn.Sequential(     #->(3,60,60)
        ###############################################################
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # ->(16,60,60)
            nn.ReLU(),  # ->(16,60,60)
            nn.MaxPool2d(kernel_size=2),  # ->(16,30,30)
        )

        # ->(16,48,48)
        ###############################################################
        self.conv2 = nn.Sequential(
            nn.Conv2d(  # ->(16,48,48)
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1),  # ->(32,30,30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # ->(32,15,15)

        )

        ###############################################################
        self.linear = nn.Sequential(
            nn.Linear(32 * 15 * 15, 256),
            nn.Tanh(),  # 激活函数
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),
            nn.Tanh()
        )

        # )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 60 * 60 * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        encoded = self.linear(x)
        decoded = self.decoder(encoded)
        return encoded, decoded







