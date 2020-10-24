import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image

'''
    Abstract:
生成对抗网络：包含一个生成器(generator)和一个判别器（discriminator）
            生成器的任务是根据输入生成图像，判别器的任务是分辨出输入的
            图像属于生成器生成结果的还是原数据集
            当判别器无法分辨时(p=0.5)，网络达到最理想状态
判别器：输入图片（对于本任务），输入1或0
生成器：输入噪声参数，输出图片
每个batch中，首先训练判别器，再训练生成器
'''

'''
    Note:
1. 要分清楚判别器和生成器的目标，以正确写出loss
2. 判别器训练方法：将训练集数据向全真靠拢，虚假数据向全假靠拢
    生成器训练方法：产生虚假数据，让判别器的判别结果向全真靠拢
    总的来说，以判别器的结果和数据来源为
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
latent_size = 64
hidden_size = 256
image_size = 784  # 28x28x1
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

# 创建文件夹
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],
                                                     std=[0.5])])

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           transform=transform,
                                           train=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True,
                                               batch_size=batch_size)

# 判别器：输入图片，输出真假
D = nn.Sequential(nn.Linear(image_size, hidden_size),
                  nn.LeakyReLU(0.2),
                  nn.Linear(hidden_size, hidden_size),
                  nn.LeakyReLU(0.2),
                  nn.Linear(hidden_size, 1),
                  nn.Sigmoid())
# 生成器：输入参数，输出图片
G = nn.Sequential(nn.Linear(latent_size, hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.ReLU(),
                  nn.Linear(hidden_size, image_size),
                  nn.Tanh())

D = D.to(device)
G = G.to(device)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


# 将输入限制到[0, 1]
def denorm(x):
    temp = (x + 1) / 2
    return temp.clamp(0, 1)


# 训练
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_dataloader):
        images = images.reshape(batch_size, -1).to(device)
        # 创建全0、全1的labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ====================训练判别器=======================#
        # 使用训练集数据训练，因为图片全部来自训练集，所以都为真
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)  # 降低结果和全真的距离
        # 使用虚假数据进行训练
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)  # 降低结果和全假的距离，即让结果更接近全假
        # 进行反向传播和优化
        # 优化的目标是，让它判断错误的概率降低
        d_total_loss = d_loss_real + d_loss_fake
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        d_total_loss.backward()
        d_optimizer.step()

        # ====================训练生成器=======================#
        # 生成器的最终目的是，让判别器认为它生成的图片是原训练集上的
        # 所以优化的目标是，判别器认为它是虚假的概率降低
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        g_loss = criterion(D(fake_images), real_labels)  # 降低结果距全真的距离，即让结果更接近全真
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 100 == 0:
            print("epoch:{}\tbatch:{}\tdloss:{}\tgloss:{}".format(epoch+1, i+1, d_total_loss, g_loss))

    if (epoch+1) ==1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images-{}.png'.format(epoch+1)))
    fake_images = fake_images.reshape(images.size(0), 1, 28, 28)
    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))