import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import cv2
import os
from torch.utils.tensorboard import SummaryWriter
from model import AutoEncoder

tb = SummaryWriter()
# 定义loss
def loss_function(recon_x, x, mu, logvar):
    print(recon_x.shape)
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print(BCE+KLD)
    return BCE+KLD

# 训练并反向传播
def trainOneBatch(batch: torch.FloatTensor, raw: torch.FloatTensor):
    decoded, mu, log_var = auto.forward(batch)
    tb.add_graph(auto, batch)
    loss = loss_function(decoded, raw, mu, log_var)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 前向传播获得误差
def testOneBatch(batch: torch.FloatTensor, raw: torch.FloatTensor):
    decoded, mu, log_var = auto.forward(batch)
    loss = loss_function(decoded, raw, mu, log_var)
    return loss

# 加载数据集
def getTrainData(data_path="./celeba_select/"):
    files = os.listdir(data_path)
    imgs = []  # 构造一个存放图片的列表数据结构
    for file in files:
        file_path = data_path + "/" + file
        img = cv2.imread(file_path)
        if img is None:
            print('错误! 无法在该地址找到图片!')
        dim = (64, 64)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.005, help='lr')
    parser.add_argument('--batch_size', type=int, default=144, help='batch size')
    parser.add_argument('--epoch', type=int, default=30, help='epoch size')
    opt = parser.parse_args()
    # 超参数
    LR = opt.lr
    BATCH_SIZE = opt.batch_size
    EPOCHES = opt.epoch
    # 获取gpu是不是可用
    cuda_available = torch.cuda.is_available()
    # 实例化网络
    auto = AutoEncoder()
    if cuda_available:
        auto.cuda()
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(auto.parameters(), lr=LR)
    # 数据准备
    imgs = getTrainData()
    for i in range(EPOCHES):
        # 打乱数据
        np.random.shuffle(imgs)
        count = 0  # count是为了凑齐成为一个batch_size的大小
        batch = []
        for j in range(len(imgs)):
            img = imgs[j]
            count += 1
            batch.append(img)
            if count == BATCH_SIZE or j == len(imgs) - 1:  # 这里就算最后
                # 列表转成张量，再转换维度
                batch_train = torch.Tensor(batch).permute(0, 3, 2, 1) / 255  # batch,3,64,64
                raw = batch_train.contiguous().view(batch_train.size(0), -1)  # batch,3*64*64
                print(batch_train.shape)
                print(raw.shape)
                if cuda_available:
                    raw = raw.cuda()  # 数据变换到gpu上
                    batch_train = batch_train.cuda()
                trainOneBatch(batch_train, raw)  # 训练一个批次
                batch.clear()
                count = 0
        batch.clear()
        # 测试
        for j in range(100):
            batch.append(imgs[j])
            batch_train = torch.Tensor(batch).permute(0, 3, 2, 1)/255
            raw = batch_train.contiguous().view(batch_train.size(0), -1)
            if cuda_available:
                raw = raw.cuda()
                batch_train = batch_train.cuda()
        # 调用函数获得损失
        loss = testOneBatch(batch_train, raw)
        batch.clear()
        #tb.add_scalar('Loss', loss, i)
        print(loss)
        # 把训练的中间结果输出到本地文件
        torch.save(auto, "autoencoder.pkl")
