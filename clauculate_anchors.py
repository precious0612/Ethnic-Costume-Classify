# -*- coding: utf-8 -*-
# 根据标签文件求先验框

import os
import numpy as np
import xml.etree.cElementTree as et
from kmeans import kmeans, avg_iou

FILE_ROOT = "/Users/precious/Classification_of_ethnic_costumes/yolo_data/"     # 根路径
ANNOTATION_ROOT = "Annotations"  # 数据集标签文件夹路径
ANNOTATION_PATH = FILE_ROOT + ANNOTATION_ROOT

ANCHORS_TXT_PATH = "/Users/precious/Classification_of_ethnic_costumes/yolov5/data/anchors.txt"

CLUSTERS = 9
CLASS_NAMES = ['achang', 'baizu']

def load_data(anno_dir, class_names):
    xml_names = os.listdir(anno_dir)
    # print(xml_names)
    boxes = []
    for xml_name in xml_names:
        xml_pth = os.path.join(anno_dir, xml_name)
        # tree = et.parse(xml_pth)
        try:
            with open(xml_pth, 'r', encoding='utf-8') as box_file:
                # print(box_file)
                objects = box_file.readlines()
        except:
            os.remove(anno_dir+"/.DS_Store")
            with open(xml_pth, 'r', encoding='utf-8') as box_file:
                objects = box_file.readlines()[:-1]

        # width = float(tree.findtext("./size/width"))
        # height = float(tree.findtext("./size/height"))

        # print(objects)

        for obj in objects:
            # print(obj)
            class_num, x_center, y_center, width, height = obj.split(' ')
            class_num = eval(class_num)
            x_center = eval(x_center)
            y_center = eval(y_center)
            width = eval(width)
            height = eval(height)
            if class_num in range(len(class_names)):
                # _, x_center, y_center, width, height = obj.split(' ')
                xmin = x_center - width / 2
                ymin = y_center - height / 2
                xmax = x_center + width / 2
                ymax = y_center + width / 2

                box = [xmax - xmin, ymax - ymin]
                boxes.append(box)
                print(boxes)
            else:
                continue
    return np.array(boxes)

if __name__ == '__main__':

    anchors_txt = open(ANCHORS_TXT_PATH, "w")

    train_boxes = load_data(ANNOTATION_PATH, CLASS_NAMES)
    # print(train_boxes)
    count = 1
    best_accuracy = 0
    best_anchors = []
    best_ratios = []

    for i in range(10):      ##### 可以修改，不要太大，否则时间很长
        anchors_tmp = []
        clusters = kmeans(train_boxes, k=CLUSTERS)
        idx = clusters[:, 0].argsort()
        clusters = clusters[idx]
        # print(clusters)

        for j in range(CLUSTERS):
            anchor = [round(clusters[j][0] * 640, 2), round(clusters[j][1] * 640, 2)]
            anchors_tmp.append(anchor)
            print(f"Anchors:{anchor}")

        temp_accuracy = avg_iou(train_boxes, clusters) * 100
        print("Train_Accuracy:{:.2f}%".format(temp_accuracy))

        ratios = np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist()
        ratios.sort()
        print("Ratios:{}".format(ratios))
        print(20 * "*" + " {} ".format(count) + 20 * "*")

        count += 1

        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_anchors = anchors_tmp
            best_ratios = ratios

    anchors_txt.write("Best Accuracy = " + str(round(best_accuracy, 2)) + '%' + "\r\n")
    anchors_txt.write("Best Anchors = " + str(best_anchors) + "\r\n")
    anchors_txt.write("Best Ratios = " + str(best_ratios))
    anchors_txt.close()
