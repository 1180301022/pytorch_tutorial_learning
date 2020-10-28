import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

'''
    Abstract:
实现了变分自编码器VAE

自编码器AE：
    给定输入，提取输入特征，并输出还原后的输入
    AE无法无输入有效生成，无法构造隐藏向量，
    因为是通过一张图片输入编码得到隐向量的。
    
变分自编码器VAE：
    在编码过程给它增加一些限制，迫使其生成的
    隐含向量能够粗略的遵循一个标准正态分布。
    每一个样本，都有其专属的正态分布。
    
    VAE本质上就是在AE的基础上，对 encoder1 的结果（计算均值的网络）加上了“高斯噪声”，
    使得结果 decoder 能够对噪声有鲁棒性；
    encoder2（计算方差的网络）的作用是用来动态调节噪声的强度。
'''

'''
    Note:
1. 重参数：
    “采样”这个操作是不可导的，而采样的结果是可导的
    将从N(μ,σ2)采样变成了从N(0,1)中采样，
    然后通过参数变换得到从N(μ,σ2)中采样的结果。
    这样一来，“采样”这个操作就不用参与梯度下降了，
    改为采样的结果参与，使得整个模型可训练了。
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

image_size = 784  # 28x28
h_dim = 400
z_dim = 20  # 每个样本的高斯分布的维数
num_epochs = 5
batch_size = 128
learning_rate = 1e-3

dataset = torchvision.datasets.MNIST(root='../../data',
                                     transform=transforms.ToTensor(),
                                     train=True)
dataLoader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=True,
                                         batch_size=batch_size)


class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        # fc1~3是encoder，4和5是decoder
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)

    # 编码部分，计算均值和方差
    def encode(self, x):
        h = F.relu(self.fc1(x))
        encoded1 = self.fc2(h)
        encoded2 = self.fc3(h)
        return encoded1, encoded2

    # 重参化: 从N(μ,σ2)中采样一个Z，相当于从N(0,1)中采样一个ε，然后让Z=μ+ε×σ
    # 每个样本对应一个 多维 正态分布，所以每个样本都有自己的均值和方差
    # 返回每个正态分布中的取样结果
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 译码部分
    def decode(self, z):
        h = F.relu(self.fc4(z))
        decoded = F.sigmoid(self.fc5(h))
        return decoded

    def forward(self, x):
        # encode
        mu, log_var = self.encode(x)
        # latent采样
        z = self.reparameterize(mu, log_var)
        # decode
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(dataLoader):
        images = images.reshape(-1, image_size).to(device)
        x_reconst, mu, log_var = model(images)
        # 计算重构损失、KL散度
        re_loss = F.binary_cross_entropy(x_reconst, images, size_average=False)
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = re_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print('epoch:{}\tbatch:{}\tloss:{}\t'.format(epoch+1, i+1, loss.item()))

    with torch.no_grad():
        # 保存sample
        # encode后的结果随机时的产生结果
        z = torch.randn(batch_size, z_dim).to(device)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sample-{}.png'.format(epoch+1)))
        # 保存重构图片
        # 输入原图片，保存生成图片和原图片的对比
        out, _, _ = model(images)
        x_concat = torch.cat([images.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)  # 拼接图片，左侧原图
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))