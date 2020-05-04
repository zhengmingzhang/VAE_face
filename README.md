## AutoEncoder简介
**autoencoder**就是自动编码器，它的原理就不做介绍了网上有很多，简单的说就是一种自学习的无监督学习算法，它的输入与输出相同。我理解就是autoencoder能够学习数据的内部特征，因此这个算法常常被用来做数据降维、特征筛选和图像去噪音。我最开始接触到自动编码器是在ABB中国研究院实习的时候，当时带我的博士让我做一个基于机器学习的齿轮箱故障检测的项目，我就是用到了自动编码器来对震动数据的频谱进行降维。我的硕士论文也用到了autoencoder，说起来和这个算法还是有些缘分的。
最近working from home，看了蛮多的有趣的开源项目，发现autoencoder的应用还是非常广泛的，之前很火的DeepFake的基本原理竟然也是基于autoencoder，因此我想自己尝试训练一个autoencoder对人脸进行一个生成任务

## VAE
尝试了简单的AE来生成随机的人脸，但是效果不是很行，查找相关研究找到了变分自编码（VAE），效果还不错。

## 如何使用
将celeba下载到文件夹中，我这里只使用了celeba数据集的前5000个照片。

模型训练
~~~python
python train.py
~~~

检验模型效果
~~~python
python API.py
~~~

## 模型效果

reconstruction

![reconstruction](https://github.com/zhengmingzhang/VAE_face/blob/master/imgs/reconstruction.png)

sample generate

![generate](https://github.com/zhengmingzhang/VAE_face/blob/master/imgs/53.png)
