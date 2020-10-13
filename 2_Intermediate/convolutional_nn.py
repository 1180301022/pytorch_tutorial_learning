import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

'''
    Abstract:
使用双卷积层和一个全连接层的图像多分类任务
conv->bn->relu->pooling
'''

'''
    Note:
1. torch.nn.Sequential 可以包含一组操作，即函数集合，
   等价于在forward中进行多步操作，可以脱离model使用
2. forward中需要返回计算后的x
3. model.eval()会改变模型中batch_norm的行为，用于测试模型
4. enumerate() 函数用于将一个可遍历的数据对象(如列表、元组
    或字符串)组合为一个索引序列，同时列出数据和数据下标
'''

# 设置设备和超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
num_classes = 10
batch_size = 100
alpha = 1e-3

# 读取数据
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


class ConvNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNN, self).__init__()
        self.layer1 = nn.Sequential(  # nn.Sequential定义一组操作
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 1 channel -> 16 channel
            nn.BatchNorm2d(16),  # 和channel相对应
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),  # 和channel相对应
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)  # 全连接层

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)  # (n, w, h, c) -> (n, m)
        x = self.fc(x)
        return x  # 记得返回！！


# 创建模型、指定cost function 和 optim
model = ConvNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

# 训练模型
for epoch in range(0, num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        # fp
        y_pred = model(images)
        loss = criterion(y_pred, labels)
        # bp & optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('epoch:{}\tbatch:{}\tloss:{}'.format(epoch+1, i, loss))

# 测试模型
# 评估模式：batch_norm的方式改变
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for _, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, prediction = output.max(1)
        total += labels.size(0)
        for i in range(0, len(prediction)):
            if prediction[i] == labels[i]:
                correct += 1
    print('accuracy:{}'.format(correct/total))