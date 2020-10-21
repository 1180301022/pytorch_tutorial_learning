import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

'''
    Abstract:
任务是双向LSTM做手写数字分类
双向RNN/LSTM不仅考虑到当前输入之前的序列，还考虑到之后的序列（完形填空）
所以隐藏层数目加倍（在创建LSTM时不需显式加倍，而将bidirectional=True即可）
因为隐藏层数目加倍，所以得到的输出加倍，因为一个隐藏层对应一个输出
'''

'''
    Note:
1. 学习了双向RNN和单向相比的参数差距
2. tensor.item()可以取出单元素张量的具体值，用于数值运算
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sequence_length = 28  # 底部的序列长度
input_size = 28   # 每个单元输入图像的一列
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 1
learning_rate = 0.003

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


class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True)  # 指定双向LSTM，不需说明序列长度
        self.fc = nn.Linear(2*hidden_size, num_classes)  # BiRNN具有2*hidden_size的隐层数目

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)  # 因为是双向所以h0的参数第一维加倍
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out.size()= batch_size, seq_length, hidden_size*2
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的seq input的输出结果
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        prediction = model(images)
        loss = criterion(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print("epoch:{}\tbatch:{}\tloss:{}".format(epoch+1, i+1, loss))

model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in test_dataloader:
        images = images.reshape(batch_size, sequence_length, input_size).to(device)
        labels = labels.reshape(batch_size, -1).to(device)
        output = model(images)
        labels = labels
        _, output = output.max(1)
        output = output.reshape(batch_size, 1)
        total += output.size(0)
        correct += torch.sum(output == labels).item()
    accuracy = correct / total
    print(accuracy)