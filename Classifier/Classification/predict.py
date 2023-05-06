import numpy as np
import torch
from PIL import Image
import transforms

from nn_model import nets

image = input('请输入图片路径：')
img = Image.open(image)
# img.show()
img = img.convert('RGB')  # png四通道,jpg三通道
# trans = transforms.Compose([
#     transforms.ToTensor()])
trans = transforms.get_test_transform()
img = trans(img)
img = img.reshape((1, )+img.size())

model = nets('resnet18')
model.load_state_dict(torch.load('./models/10.pth', map_location=torch.device('cpu')))  # 将gpu训练的模型在cpu上跑
print('------模型已加载完毕-------')
classes = ['achang', 'bai', 'baoan', 'bulang', 'buyi', 'chaoxain', 'dai', 'dawoer', 'deang',
           'dong', 'dongxiang', 'dulong', 'elunchun', 'eluosi', 'ewenke', 'gaoshan', 'gelao', 'hani',
           'han', 'hasake', 'hezhe', 'hui', 'jing', 'jingpo', 'jinuo', 'keerkezi', 'lahu',
           'li', 'luoba', 'man', 'maonan', 'menba', 'menggu', 'miao', 'mulao', 'naxi', 'nu',
           'pumi', 'qiang', 'sala', 'she', 'shui', 'susu', 'tajike', 'tataer', 'tujia', 'tu',
           'wa', 'weiwuer', 'wuzibieke', 'xibo', 'yao', 'yi', 'yugu', 'zang', 'zhaung'
           ]
model.eval() 
with torch.no_grad():
    print(img)
    exit()
    output = model(img)
    class_pre = output.argmax(1).item()  # 预测类
    print(f'预测结果为：{classes[class_pre]}, zu')


