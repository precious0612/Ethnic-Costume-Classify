import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from Dataset import train_dataloader, test_dataloader, len_test, len_train
from nn_model import nets

device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = train_dataloader
test_data = test_dataloader
train_data_size = len_train
test_data_size = len_test
print('训练数据集长度：{}'.format(train_data_size))
print('测试数据集长度：{}'.format(test_data_size))

model = nets('resnet18')
model = model.to(device_gpu)
lr = 1e-2  # 0.01

# 损失函数
loss = nn.CrossEntropyLoss()
loss = loss.to(device_gpu)
# 优化器
optim = torch.optim.SGD(model.parameters(), lr=lr)

train_step = 0
test_step = 0
epoch = 100

writer = SummaryWriter('../logs')
for i in range(epoch):
    print(f'--------第{i+1}轮训练开始--------')
    model.train()  
    for data in train_data:
        imgs, train_labels = data
        imgs = imgs.to(device_gpu)
        train_labels = train_labels.to(device_gpu)
        output = model(imgs)
        loss_ = loss(output, train_labels.long())  # 要求输入为long
        optim.zero_grad()
        loss_.backward()
        optim.step()
        train_step = train_step + 1
        if train_step % 20 == 0:
            print('训练次数:{}, Loss:{}'.format(train_step, loss_.item()))  
            writer.add_scalar('train_loss', loss_.item(), train_step)
# 每个epoch测试一次损失
    model.eval()  
    total_acc = 0
    loss_sum = 0
    acc_sum = 0
    with torch.no_grad():
        for data in test_data:
            imgs, test_labels = data
            imgs = imgs.to(device_gpu)
            test_labels = test_labels.to(device_gpu)
            output = model(imgs)
            loss_ = loss(output, test_labels.long())
            loss_sum = loss_sum + loss_
            acc = (output.argmax(1) == test_labels).sum()
            acc_sum = acc_sum + acc
            total_acc = total_acc + acc
    print(f'正确个数:{acc_sum},总数：{test_data_size}')
    print('测试集整体损失：{}'.format(loss_sum))
    print('测试集整体正确率：{}'.format(total_acc / test_data_size))
    writer.add_scalar('test_loss', loss_sum, test_step)
    writer.add_scalar('test_acc', total_acc, test_step)
    test_step = test_step + 1

# 保存模型
    if (total_acc / test_data_size) > 0.25:
        torch.save(model.state_dict(), 'models/{}.pth'.format(i+1))
        print(f'--------第{i+1}轮模型已保存--------')


writer.close()
