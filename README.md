# 民族服装图像分类和检测项目

[English Version](README.en.md)|[中文版本](README.md)

## 概览

该项目始于2022年12月，作为一个机器学习课程设计实验。项目利用机器学习算法从策划的图像数据集中分类和检测民族服装。

该项目是在MacBook Pro 2019上开发的，规格包括Intel Core i9-9880H 2.3GHz (4.8GHz Turbo Boost)，8GB AMD Radeon Pro 5500M，运行macOS 13 Ventura。

项目的工作流程分为两个主要部分：

1. **图像分类任务**
2. **目标检测任务**

## 第一部分：图像分类任务

这个任务使用Tensorflow（Keras）和PlaidML（Keras）实现的ResNet18模型。此任务的相关文件包括：

- `tensorflow.ipynb`：在Tensorflow（Keras）中实现的ResNet18分类任务。
- `test.ipynb`：在PlaidML（Keras）中实现的ResNet18分类任务。

分类任务使用的数据集分为两个子文件夹，`train` 和 `test`，分别包含2609张和1160张图片。

| Ethnic Group | Number of Training Images | Number of Test Images |
| --- | --- | --- |
| dong | 30 | 13 |
| gelaozu | 46 | 14 |
| hanzu | 33 | 18 |
| tataer | 43 | 20 |
| hezhezu | 29 | 20 |
| lahu | 41 | 20 |
| jinuo | 43 | 20 |
| yizu | 38 | 20 |
| elunchunzu | 38 | 20 |
| miaozu | 38 | 20 |
| baoan | 42 | 20 |
| menbazu | 45 | 20 |
| manzu | 36 | 20 |
| pumi | 40 | 20 |
| salazu | 35 | 20 |
| she | 38 | 20 |
| eluosizu | 42 | 20 |
| nuzu | 50 | 20 |
| chaoxian | 50 | 20 |
| shuizu | 50 | 20 |
| deangzu | 50 | 20 |
| hasakezu | 50 | 20 |
| dawoer | 50 | 20 |
| bulangzu | 50 | 20 |
| susu | 50 | 20 |
| ewenkezu | 50 | 20 |
| mulaozu | 50 | 20 |
| maonanzu | 50 | 20 |
| weiwuezu | 50 | 20 |
| baizu | 50 | 20 |
| keerkezizu | 50 | 20 |
| yugu | 50 | 20 |
| tuzu | 50 | 20 |
| tajike | 50 | 20 |
| wuzibiekezu | 50 | 20 |
| qiangzu | 50 | 20 |
| lizu | 50 | 20 |
| yao | 50 | 20 |
| dongxiangzu | 50 | 20 |
| menguzu | 50 | 20 |
| daizu | 50 | 20 |
| hanizu | 50 | 20 |
| gaoshan | 50 | 20 |
| naxizu | 50 | 20 |
| jing | 50 | 20 |
| luobazu | 50 | 20 |
| xibozu | 50 | 20 |
| tujiazu | 50 | 20 |
| zangzu | 50 | 20 |
| zhuang | 50 | 20 |
| huizu | 50 | 20 |
| buyi | 50 | 20 |
| jingpozu | 50 | 20 |
| dulongzu | 50 | 20 |
| achange | 51 | 20 |
| wa | 50 | 20 |
| **Total number of files** | **2609** | **1106** |

## 第二部分：目标检测任务

这个任务基于YOLOv5模型。最初使用的数据集与分类任务相同。然而，由于民族服装之间的差异微妙，我们创建了一个调整后的数据集（`images_changed`），其中每张图片的服装都被单独分开。尽管进行了进一步的数据筛选和处理，但最后的实验结果并未达到我们的预期。所有训练过程曲线都保存在`yolo_line`文件夹中。

| Ethnic Group | Number of Training Images | Number of Test Images | Number of Changed Training Images | Number of Changed Test Images |
| --- | --- | --- | --- | --- |
| dong | 30 | 13 | 50 | 47 |
| gelaozu | 46 | 14 | 49 | 45 |
| hanzu | 33 | 18 | 60 | 40 |
| tataer | 43 | 20 | 69 | 41 |
| hezhezu | 29 | 20 | 35 | 37 |
| lahu | 41 | 20 | 48 | 33 |
| jinuo | 43 | 20 | 44 | 37 |
| yizu | 38 | 20 | 72 | 44 |
| elunchunzu | 38 | 20 | 41 | 31 |
| miaozu | 38 | 20 | 51 | 52 |
| baoan | 42 | 20 | 54 | 42 |
| menbazu | 45 | 20 | 48 | 38 |
| manzu | 36 | 20 | 48 | 46 |
| pumi | 40 | 20 | 49 | 46 |
| salazu | 35 | 20 | 48 | 37 |
| she | 38 | 20 | 48 | 37 |
| eluosizu | 42 | 20 | 68 | 53 |
| nuzu | 50 | 20 | 45 | 35 |
| chaoxian | 50 | 20 | 60 | 50 |
| shuizu | 50 | 20 | 42 | 35 |
| deangzu | 50 | 20 | 47 | 33 |
| hasakezu | 50 | 20 | 51 | 43 |
| dawoer | 50 | 20 | 65 | 30 |
| bulangzu | 50 | 20 | 47 | 36 |
| susu | 50 | 20 | 57 | 66 |
| ewenkezu | 50 | 20 | 43 | 34 |
| mulaozu | 50 | 20 | 60 | 52 |
| maonanzu | 50 | 20 | 48 | 35 |
| weiwuezu | 50 | 20 | 60 | 50 |
| baizu | 50 | 20 | 54 | 55 |
| keerkezizu | 50 | 20 | 48 | 35 |
| yugu | 50 | 20 | 35 | 37 |
| tuzu | 50 | 20 | 45 | 37 |
| tajike | 50 | 20 | 57 | 40 |
| wuzibiekezu | 50 | 20 | 55 | 53 |
| qiangzu | 50 | 20 | 48 | 39 |
| lizu | 50 | 20 | 52 | 36 |
| yao | 50 | 20 | 45 | 36 |
| dongxiangzu | 50 | 20 | 54 | 46 |
| menguzu | 50 | 20 | 62 | 51 |
| daizu | 50 | 20 | 70 | 68 |
| hanizu | 50 | 20 | 48 | 51 |
| gaoshan | 50 | 20 | 52 | 40 |
| naxizu | 50 | 20 | 48 | 40 |
| jing | 50 | 20 | 58 | 53 |
| luobazu | 50 | 20 | 38 | 39 |
| xibozu | 50 | 20 | 43 | 32 |
| tujiazu | 50 | 20 | 66 | 39 |
| zangzu | 50 | 20 | 55 | 40 |
| zhuang | 50 | 20 | 66 | 42 |
| huizu | 50 | 20 | 61 | 43 |
| buyi | 50 | 20 | 60 | 42 |
| jingpozu | 50 | 20 | 49 | 34 |
| dulongzu | 50 | 20 | 42 | 31 |
| achange | 51 | 20 | 48 | 47 |
| wa | 50 | 20 | 50 | 48 |
| **Total number of files** | **2609** | **1106** | **2917** | **2360** |

这个任务的项目结构包括：

- `yolo_data`：保存YOLO格式的数据集，用于训练。
- `yolo_line`：保存训练曲线。
- `yolov5`：包含YOLOv5项目库和为训练进行的项目修改。
