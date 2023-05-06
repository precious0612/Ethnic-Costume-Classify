import numpy as np
import torch
from Classification.transforms import get_train_transform, get_test_transform
from PIL import Image
from torch.utils.data import Dataset, DataLoader

input_size = 300
batch_size = 20


class SelfCustomDataset(Dataset):
    def __init__(self, label_file, imageset):
        super(SelfCustomDataset, self).__init__()  # 继承torch中的Dataset类
        self.img_aug = True
        with open(label_file, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
            # 打开图片文件夹，去除换行符，以空格作为分隔生成列表
            # [['../images/test\\achang\\1.jpg', '0'], ... ]
        if imageset == 'train':
            self.transform = get_train_transform(size=input_size)
        else:
            self.transform = get_test_transform(size=input_size)
        self.input_size = input_size

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        # 保持图片格式为三通道RGB格式
        if self.img_aug:
            img = self.transform(img)
        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))

    def __len__(self):
        return len(self.imgs)


train_label_dir = '../Classification/train.txt'
train_datasets = SelfCustomDataset(train_label_dir, imageset='train')
len_train = len(train_datasets)
train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

test_label_dir = '../Classification/test.txt'
test_datasets = SelfCustomDataset(test_label_dir, imageset='test')
len_test = len(test_datasets)
test_dataloader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

# 进行数据提取函数的测试
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in test_dataloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        print(images[:, 1, :, :])  # torch.Size([2, 3, 300, 300])





