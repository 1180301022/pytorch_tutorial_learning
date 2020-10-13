import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

'''
    Abstract:
MNIST数据集上的多分类问题
使用自定前馈神经网络：linear->relu->linear
使用GPU计算
'''

'''
    Note:
1. 使用GPU计算时，首先设置设备 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    在获取数据、创建模型时，在操作后加 .to(device)，将其转移到GPU
    创建dataset、dataloader时不需要
2. 测试时需要指定 with torch.no_grad()，否则会创建计算图，造成不必要运算
3. 自定模型时注意__init__的格式
'''

# 设备选择 和 超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 28 * 28
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
alpha = 1e-3

# 加载数据集
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)


class SingleHiddenLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SingleHiddenLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 创建模型，指定代价函数和优化方式
model = SingleHiddenLayerNN(input_size, hidden_size, num_classes).to(device)  # 转移到GPU
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=alpha)  # 使用Adam优化方法

# 进行训练
for epoch in range(0, num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, input_size).to(device)  # 使用GPU
        labels = labels.to(device)
        # fp
        pred = model(images)
        loss = criterion(pred, labels)
        # bp & optim
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print
        if i % 100 == 0:
            print('epoch:{}\tbatch:{}\tloss:{}'.format(epoch + 1, i, loss.data))

# 测试
with torch.no_grad():
    total = 0
    accurate = 0
    for images, labels in train_dataloader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        output = model(images)
        _, prediction = torch.max(output, 1)
        total += labels.size(0)
        for i in range(0, labels.size(0)):
            if torch.equal(prediction[i], labels[i]):
                accurate += 1
    print('accuracy:{}'.format(accurate / total))
