import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import cv2
import os
from dataload import *
from model import AutoEncoder
from torchvision import datasets, transforms

# 定义loss
def loss_function(recon_x, x, mu, logvar):
    # how well do input x and output recon_x agree?
    BCE = 0
    for recon_x_one in recon_x:
        BCE += F.binary_cross_entropy(recon_x_one, x.view(-1, 3 * 64 * 64))
    BCE /= len(recon_x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * 3 * 64 * 64

    return BCE + KLD

# 训练并反向传播
def trainOneBatch(batch: torch.FloatTensor, raw: torch.FloatTensor):
    decoded, mu, log_var = auto.forward(batch)
    loss = loss_function(decoded, raw, mu, log_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 前向传播获得误差
def testOneBatch(batch: torch.FloatTensor, raw: torch.FloatTensor):
    decoded, mu, log_var = auto.forward(batch)
    loss = loss_function(decoded, raw, mu, log_var)
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005, help='lr')
    parser.add_argument('--batch_size', type=int, default=144, help='batch size')
    parser.add_argument('--epoch', type=int, default=1, help='epoch size')
    opt = parser.parse_args()
    # 超参数
    LR = opt.lr
    BATCH_SIZE = opt.batch_size
    EPOCHES = opt.epoch
    LOG_INTERVAL = 5
    # 获取gpu是不是可用
    cuda_available = torch.cuda.is_available()
    # 实例化网络
    auto = AutoEncoder()
    if cuda_available:
        auto.cuda()
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(auto.parameters(), lr=LR)
    # 数据准备
    root_dir = "./celeba_select"
    image_files = os.listdir(root_dir)
    train_dataset = CelebaDataset(root_dir, image_files, (64, 64), transforms.Compose([ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=1, shuffle=True)
    for i in range(EPOCHES):
        # 打乱数据
        auto.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = Variable(data.type(torch.FloatTensor))
            if cuda_available:
                data = data.cuda()
            optimizer.zero_grad()

            # push whole batch of data through VAE.forward() to get recon_loss
            recon_batch, mu, logvar = auto(data)
            # calculate scalar loss
            loss = loss_function(recon_batch, data, mu, logvar)
            # calculate the gradient of the loss w.r.t. the graph leaves
            # i.e. input variables -- by the power of pytorch!
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(i, train_loss / len(train_loader.dataset)))

        torch.save(auto, "autoencoder.pkl")
