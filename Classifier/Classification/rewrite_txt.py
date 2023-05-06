import glob
import os
import random

if __name__ == '__main__':
    train_data_path = '../../images/train'
    test_data_path = '../../images/test'
    train_labels = os.listdir(train_data_path)
    test_labels = os.listdir(test_data_path)
    # 写train.txt文件
    txt_path = 'Classification'
    for index, label in enumerate(train_labels):
        train_img_list = glob.glob(os.path.join(train_data_path, label, '*.jpg'))
        random.shuffle(train_img_list)
        # 划分测试集与训练集
        # train_list = train_img_list[:int(0.8*len(img_list))]
        # testlist = train_img_list[(int(0.8*len(img_list)):]
        with open('./train.txt', 'a')as f:
            for img in train_img_list:
                f.write(img + ' ' + str(index))
                f.write('\n')

    # 写test.txt文件
    for index, label in enumerate(test_labels):
        test_img_list = glob.glob(os.path.join(test_data_path, label, '*.jpg'))
        random.shuffle(test_img_list)
        with open('./test.txt', 'a')as f:
            for img in test_img_list:
                f.write(img + ' ' + str(index))
                f.write('\n')


