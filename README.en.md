# Ethnic Clothing Image Classification and Detection Project

[中文版本](README.md)|[English Version](README.en.md)

## Overview

This project was initiated in December 2022 as part of a Machine Learning course design experiment. The project utilizes machine learning algorithms to classify and detect ethnic clothing from a curated image dataset. 

The project was developed on a MacBook Pro 2019, with specifications including an Intel Core i9-9880H 2.3GHz (4.8GHz Turbo Boost), 8GB AMD Radeon Pro 5500M, running macOS 13 Ventura.

The project's workflow is divided into two main sections: 

1. **Image Classification Task**
2. **Target Detection Task**

## Part 1: Image Classification Task

This task utilizes the ResNet18 model implemented in Tensorflow (Keras) and PlaidML (Keras). The relevant files for this task include:

- `tensorflow.ipynb`: ResNet18 classification task implemented in Tensorflow (Keras).
- `test.ipynb`: ResNet18 classification task implemented in PlaidML (Keras).

The classification task employs a dataset divided into two sub-folders, `train` and `test`, with 2609 and 1160 images respectively.


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

## Part 2: Target Detection Task

This task is based on the YOLOv5 model. The initial dataset used is the same as in the classification task. However, due to the subtle differences between ethnic costumes, we created an adjusted dataset (`images_changed`) where costumes in each picture are split separately. Despite further data screening and processing, the final experimental results did not meet our expectations. All the training process curves are saved in the `yolo_line` folder.

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

The project structure for this task includes:

- `yolo_data`: Holds the YOLO-formatted dataset, used for training.
- `yolo_line`: Stores the training curves.
- `yolov5`: Contains the YOLOv5 project repository and the modifications made to the project for training.
