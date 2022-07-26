# CutPaste-paddle

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
- [5. 模型推理部署]()
    - [5.1 基于Inference的推理]()
    - [5.2 基于Serving的服务化部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()

## 1. 简介
cutpaste是一种简单有效的自监督学习方法，其目标是构建一个高性能的两阶段缺陷检测模型，在没有异常数据的情况下检测图像的未知异常模式。首先通过cutpaste数据增强方法学习自监督深度表示，然后在学习的表示上构建生成的单类分类器。

**论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)

在此非常感谢`Runinho`等人贡献的[pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste) ，提高了本repo复现论文的效率。

**aistudio体验教程:** [CutPaste_paddle](https://aistudio.baidu.com/aistudio/projectdetail/4378924?contributionType=1&shared=1)


## 2. 数据集和复现精度

- 数据集大小：共包含15个物品类别，解压后总大小在4.92G左右
- 数据集下载链接：[mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
# 复现精度（Comparison to Li et al.）
| defect_type   |    CutPaste (3-way) | Runinho. CutPaste (3-way) | Li et al. CutPaste (3-way) |
|:--------------|--------------------:|-------------------:|-----------------------------:|
| bottle        |                98.0 |               99.6 |                         98.3 |
| cable         |                78.8 |               77.2 |                         80.6 |
| capsule       |                95.3 |               92.4 |                         96.2 |
| carpet        |                94.6 |               60.1 |                         93.1 |
| grid          |                95.5 |              100.0 |                         99.9 |
| hazelnut      |                96.7 |               86.8 |                         97.3 |
| leather       |               100.0 |              100.0 |                        100.0 |
| metal_nut     |                97.9 |               87.8 |                         99.3 |
| pill          |                85.8 |               91.7 |                         92.4 |
| screw         |                83.7 |               86.8 |                         86.3 |
| tile          |                89.4 |               97.2 |                         93.4 |
| toothbrush    |                96.7 |               94.7 |                         98.3 |
| transistor    |                91.1 |               93.0 |                         95.5 |
| wood          |                98.7 |               99.4 |                         98.6 |
| zipper        |                99.5 |               98.8 |                         99.4 |
| average       |                93.4 |               91.0 |                         95.2 |


## 3. 准备数据与环境


### 3.1 准备环境

首先介绍下支持的硬件和框架版本等环境的要求，格式如下：

- 硬件：GPU显存建议在6G以上
- 框架：
  - PaddlePaddle >= 2.2.0
- 环境配置：直接使用`pip install -r requirements.txt`安装依赖即可。

### 3.2 准备数据

- 全量数据训练：
  - 下载好 [metec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 数据集后，将其解压到 **./Data** 文件夹下
  - 运行指令`python train.py --epochs 10000 --batch_size 32`
- 少量数据训练：
  - 运行指令`python train.py --types lite_data --data_dir ../lite_data --epochs 5 --batch_size 4`


### 3.3 准备模型

- 默认使用resnet18预训练模型进行训练，如想关闭：`python train.py --epochs 10000 --batch_size 32 --no-pretrained`


## 4. 开始使用


### 4.1 模型训练

- 全量数据训练：
  - 下载好 [metec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad/) 数据集后，将其解压到 **./Data** 文件夹下
  - 运行指令`python .\tools\train.py --epochs 10000 --batch_size 32 --cuda True`
- 少量数据训练：
  - 运行指令`python train.py --types lite_data --data_dir ../lite_data --epochs 5 --batch_size 4`
- 部分训练日志如下所示：
```
> python .\tools\train.py --epochs 10000 --batch_size 32 --cuda True
Namespace(batch_size=32, cuda='True', data_dir='Data', epochs=10000, freeze_resnet=20, head_layer=1, lr=0.03, model_dir='logs', optim='sgd', pretrained=True, save_interval=500, test_epochs=-1,
type='all', variant='3way', workers=0)
using device: cuda
training bottle
loading images
loaded 209 images
epoch:1/10000 loss:1.1202 avg_reader_cost:0.26 avg_batch_cost:2.46 avg_ips:0.08
epoch:2/10000 loss:0.9271 avg_reader_cost:0.15 avg_batch_cost:1.38 avg_ips:0.04
epoch:3/10000 loss:0.7945 avg_reader_cost:0.17 avg_batch_cost:1.08 avg_ips:0.03
epoch:4/10000 loss:0.6116 avg_reader_cost:0.18 avg_batch_cost:0.92 avg_ips:0.03
epoch:5/10000 loss:0.4354 avg_reader_cost:0.19 avg_batch_cost:0.83 avg_ips:0.03
...

> python eval.py --density sklearn --cuda 1 --head_layer 2 --save_plots 0| grep AUC
bottle AUC: 0.9944444444444445
cable AUC: 0.8549475262368815
...
``` 


### 4.2 模型评估

- 模型评估：`python eval.py --cuda True`
```
> python .\tools\eval.py --cuda True
Namespace(cuda='True', data_dir='Data', density='sklearn', head_layer=1, model_dir='logs', save_plots=True, type='all')
evaluating bottle
loading model logs/bottle/10000.pdparams
loading images
loaded 209 images
[t-SNE] Computing 82 nearest neighbors...
[t-SNE] Indexed 83 samples in 0.000s...
[t-SNE] Computed neighbors for 83 samples in 0.002s...
[t-SNE] Computed conditional probabilities for sample 83 / 83
[t-SNE] Mean sigma: 0.285641
[t-SNE] KL divergence after 250 iterations with early exaggeration: 53.737068
[t-SNE] KL divergence after 500 iterations: 0.308485
using density estimation GaussianDensitySklearn
bottle AUC: 0.9746031746031746

evaluating cable
loading model logs/cable/10000.pdparams
loading images
loaded 224 images
...
``` 

### 4.3 模型预测

- 模型预测：`python predict.py --batch_size 1`
```
> python train.py --density torch --cuda 1 --head_layer 2 --save_plots 0| grep AUC
bottle AUC: 0.9944444444444445
cable AUC: 0.8549475262368815
...

> python eval.py --density sklearn --cuda 1 --head_layer 2 --save_plots 0| grep AUC
bottle AUC: 0.9944444444444445
cable AUC: 0.8549475262368815
...
``` 


## 5. 模型推理部署

如果repo中包含该功能，可以按照Inference推理、Serving服务化部署再细分各个章节，给出具体的使用方法和说明文档。


## 6. 自动化测试脚本

介绍下tipc的基本使用以及使用链接


## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献
**参考论文:** [CutPaste: Self-Supervised Learning for Anomaly Detection and Localization](https://arxiv.org/pdf/2104.04015v1.pdf)

**参考repo:** [pytorch-cutpaste](https://github.com/Runinho/pytorch-cutpaste)