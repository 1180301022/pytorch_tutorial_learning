import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

'''
    Abstract:
使用6残差块的残差网络进行分类
残差：观测值与估计值之间的差 H(x)=F(x)+x
如果F(x)和x的维度（c/w/h）不同，则需要通过升维(0 padding / 1x1 conv 需训练 )使其一致
残差网络为了解决层数过高时的梯度爆炸和梯度消失问题
残差块可以视为一个子网络，通过堆叠残差块形成残差网络
'''

'''
    Note:
1. torchvision.transforms用来做常见的图像变换
   torchvision.transforms.Compose()用于聚合一系列图像变换的操作
2. 在卷积神经网络的卷积层之后总会添加 BatchNorm2d 进行数据的归一化处理，
    这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
    参数为 num_channels
3. tensor.max()同时返回最大值和下标
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 1
batch_size = 100
lr = 1e-3

# 图像预处理
my_transform = transforms.Compose([
    transforms.Pad(4),  # 填充，对图像上下左右均填充4个pixel
    transforms.RandomHorizontalFlip(),  # 根据概率p（默认0.5）水平翻转
    transforms.RandomCrop(32),  # 依据给定size（32）随机裁剪
    transforms.ToTensor()])  # 转换为Tensor

train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=True,
                                             transform=my_transform)
test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                             train=False,
                                             transform=my_transform)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)


# 定义剩余网络块
# conv -> bn -> relu -> conv -> bn -> +x -> relu
# 输入：输入和输出通道数、步长、下采样函数
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        # conv1可能改变w，h
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)  # 对从上层网络传递下来的tensor直接进行修改，节省内存，不用多存储其他变量
        # kernel_size=3, padding=1, 卷积前后不改变w,h
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x  # 传递引用
        temp = self.conv1(x)
        temp = self.bn1(temp)
        temp = self.relu(temp)
        temp = self.conv2(temp)
        temp = self.bn2(temp)
        # 根据输入决定是否下采样
        if self.downsample:
            residual = self.downsample(x)
        temp += residual
        temp = self.relu(temp)
        return temp


# 定义剩余网络
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])  # c:16->16
        self.layer2 = self.make_layer(block, 32, layers[1], 2)  # c:16->32
        self.layer3 = self.make_layer(block, 64, layers[2], 2)  # c:32->64
        self.avg_pooling = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    # 构造resnet中的一层，可能有多个残差块
    def make_layer(self, block, output_channels, num_blocks, stride=1):
        downsample = None
        # 如果resnet中的步长不为1 或 输入频道不等于输出频道数，需要对x进行升维
        # resnet中的步长不为1，F(x)的w，h发生变化
        if (stride != 1) or (self.in_channels != output_channels):
            downsample = nn.Sequential(
                # w，h，c升维
                nn.Conv2d(self.in_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(output_channels)
            )
        layers = []
        # 该层的首个残差块需要进行downsampling
        layers.append(block(self.in_channels, output_channels, stride, downsample))
        self.in_channels = output_channels
        # 添加该层的剩余残差块，只有第一个块改变输入的c，w，h，剩余块不改变
        for i in range(1, num_blocks):
            layers.append(block(output_channels, output_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.avg_pooling(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output


# 创建6残差块的残差网络
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)


# 更新学习率
def lr_update(optimizer, lr):
    for para_group in optimizer.param_groups:
        para_group['lr'] = lr


# 训练
curr_lr = lr
for i in range(num_epochs):
    for j, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        prediction = model(images)
        loss = criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (j + 1) % 100 == 0:
            print('epoch:{}\tbatch:{}\tloss:{}\t'.format(i+1, j+1, loss))

    if (i + 1) % 20 == 0:
        curr_lr /= 3
        lr_update(optimizer, curr_lr)

# 测试
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_dataloader:
        total += labels.size(0)
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, prediction = output.max(1)
        for i in range(labels.size(0)):
            if prediction[i] == labels[i]:
                correct += 1
    print("accuracy:{}".format(correct/total))