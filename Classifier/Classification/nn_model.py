import torch
import torchvision
from nets.LeNet import net
from nets.ResNet import net1
from nets.alexnet import alexnet


def nets(type):
    if type == 'resnet18':
        print(f'使用resnet18进行训练：')
        resnet18 = torchvision.models.resnet18(progress=True)
        resnet18.load_state_dict(torch.load('./models/resnet18-5c106cde.pth'))
        resnet18.fc.add_module('add_linear', torch.nn.Linear(1000, 56))
        # print(*[(name, param.shape) for name, param in resnet18.named_parameters()])
        return resnet18
    elif type == 'lenet':
        print(f'使用lenet进行训练：')
        return net
    elif type == 'resnet':
        print(f'使用resnet进行训练：')
        return net1
    elif type == 'alexnet':
        return alexnet
    else:
        print(f'输入错误！')





