import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

'''
    Abstract:
多对一的RNN进行MNIST数据集上的图像分类
RNN用于处理输入具有序列关系的问题（前后关联）
在各时间步共享参数:在 RNN 中，每输入一步，每一层都共享参数 U,V,W
反映RNN中的每一步都在做相同的事，只是输入不同
'''

'''
    Note:
1. LSTM之前接触过，具体可以参见收藏夹，通过此方法避免梯度消失和梯度爆炸的问题
2. 每层输入hk、xk、ck，输出hk+1、xk+1、ck+1
    c是记忆细胞，每层都会选择性的忘记之前记忆的内容，并记住新的内容
3. input_size:表示输入数据单个序列*单个时间维度*上固有的长度，一张图片的行是28
    sequence_length:表示*序列本身*的长度，列为28的图片对应的steps就等于28
    每个时间步，输入一行图像
'''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 1
learnint_rate = 0.01

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


# 多对一的RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 构造LSTM需要输入大小、隐层大小、层数
        # batch_first：输入、输出是否把batch_size放在第一维
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out = batch_size x seq_length x hidden_size
        out = self.fc(out[:, -1, :])   # 为什么取 seq_length 的最后一项？
                                    # 因为直到sl的最后一项，图片才输入完成，相当于是用一句完整的话做预测，而不是断章取义
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learnint_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('epoch:{}\tbatch:{}\tloss:{}\t'.format(epoch+1, i+1, loss))

model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in test_dataloader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        output = model(images)
        total += labels.size(0)
        _, prediction = output.max(1)
        for i in range(labels.size(0)):
            if prediction[i] == labels[i]:
                correct += 1
    print('accuracy:{}'.format(correct / total))