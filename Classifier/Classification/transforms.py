from torchvision import transforms


def get_train_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize((300, 300)),  # 图片尺寸归一化
        transforms.ToTensor(),  # PIL格式转换为tensor，在神经网络中训练
        transforms.Normalize(mean=mean, std=std),
        # 使用Imagenet的均值和标准差，将3个通道的数据进行归一化
    ])


def get_test_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), size=0):
    return transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
