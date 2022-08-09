# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from paddle.vision import transforms
from paddle.io import DataLoader
import paddle
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from tools.dataset import MVTecAT
from tools.model import ProjectionNet
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from tools.cutpaste import CutPaste, cut_paste_collate_fn
from sklearn.utils import shuffle
from collections import defaultdict
from tools.density import GaussianDensitySklearn, GaussianDensityPaddle
import pandas as pd
import numpy as np
test_data_eval = None
test_transform = None
cached_type = None

def get_train_embeds(model, size, defect_type, transform, device="cuda",datadir="Data"):
    # train data / train kde
    test_data = MVTecAT(datadir, defect_type, size, transform=transform, mode="train")

    dataloader_train = DataLoader(test_data, batch_size=32,
                            shuffle=True, num_workers=0)
    train_embed = []
    with paddle.no_grad():
        for x in dataloader_train:
            embed, logit = model(x)
            train_embed.append(embed.cpu())
    train_embed = paddle.concat(train_embed)
    return train_embed


def eval_model(modelname, defect_type, device="cpu", save_plots=False, size=256, show_training_data=False, model=None,
               train_embed=None, head_layer=8, density=GaussianDensityPaddle(),data_dir = "Data",args=None):
    # 创建全局验证数据，提升验证速度
    global test_data_eval, test_transform, cached_type

    # 如果类别发生了变化，则更新验证数据集和图像预处理方式
    if test_data_eval is None or cached_type != defect_type:
        cached_type = defect_type
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((size, size)))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]))
        test_data_eval = MVTecAT(args.data_dir, defect_type, size, transform=test_transform, mode="test")
    #创建数据加载器
    dataloader_test = DataLoader(test_data_eval, batch_size=32,
                                 shuffle=False, num_workers=0)

    # 创建并加载模型
    if model is None:
        print(f"loading model {modelname}")
        head_layers = [512] * head_layer + [128]
        weights = paddle.load(str(modelname))
        classes = 3
        model = ProjectionNet(pretrained=False, head_layers=head_layers, num_classes=classes)
        model.set_dict(weights)
        model.eval()

    # 获取深度特征图（embeds）用于计算预测器
    labels = []
    embeds = []
    with paddle.no_grad():
        for x, label in dataloader_test:
            embed, logit = model(x)

            # 保存embeds和labels
            embeds.append(embed.cpu())
            labels.append(label.cpu())
    labels = paddle.concat(labels)
    embeds = paddle.concat(embeds)

    #如果没有指定训练集的特征图(train_embed)，则当场计算一次
    if train_embed is None:
        train_embed = get_train_embeds(model, size, defect_type, test_transform, device,datadir=args.data_dir)

    # 分别对训练集和验证集的特征图进行归一化，便于后续计算
    embeds = paddle.nn.functional.normalize(embeds, p=2, axis=1)
    train_embed = paddle.nn.functional.normalize(train_embed, p=2, axis=1)


    if save_plots:
        # 创建绘制图片保存文件夹
        eval_dir = Path("logs") / defect_type
        eval_dir.mkdir(parents=True, exist_ok=True)

        # 决定是否需要输出数据增强效果，建议不要开
        show_training_data = False
        if show_training_data:
            min_scale = 1.0
            # create Training Dataset and Dataloader
            after_cutpaste_transform = transforms.Compose([])
            after_cutpaste_transform.transforms.append(transforms.ToTensor())
            after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]))

            train_transform = transforms.Compose([])
            # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
            # train_transform.transforms.append(transforms.GaussianBlur(int(size/10), sigma=(0.1,2.0)))
            train_transform.transforms.append(CutPaste(transform=after_cutpaste_transform))
            # train_transform.transforms.append(transforms.ToTensor())

            train_data = MVTecAT(args.data_dir, defect_type, transform=train_transform, size=size)
            dataloader_train = DataLoader(train_data, batch_size=32,
                                          shuffle=True, num_workers=4, collate_fn=cut_paste_collate_fn,
                                          persistent_workers=True)
            # inference training data
            train_labels = []
            train_embeds = []
            with paddle.no_grad():
                for x1, x2 in dataloader_train:
                    x = paddle.concat([x1, x2], axis=0)
                    embed, logit = model(x.to(device))

                    # generate labels:
                    y = paddle.to_tensor([0, 1])
                    # y = y.repeat_interleave(x1.size(0))
                    y = paddle.repeat_interleave(y,x1.shape[0])

                    # save
                    train_embeds.append(embed.cpu())
                    train_labels.append(y)
                    # only less data
                    break
            train_labels = paddle.concat(train_labels)
            train_embeds = paddle.concat(train_embeds)

            # for tsne we encode training data as 2, and augmentet data as 3
            tsne_labels = paddle.concat([labels, train_labels + 2])
            tsne_embeds = paddle.concat([embeds, train_embeds])
        else:
            tsne_labels = labels
            tsne_embeds = embeds
        plot_tsne(tsne_labels, tsne_embeds, eval_dir / "tsne.png")
    else:
        eval_dir = Path("unused")


    #利用训练集计算分数分布，作为最终预测器
    if args.density == "paddle":
        density.fit(train_embed,"logs/%s/params.crf"%defect_type)
    else:
        density.fit(train_embed,"logs/%s/kde.crf"%defect_type)
    #计算训练集上的分数分布，确定正常分数范围（因为训练集全是正常值，他的范围可以视为正常范围）
    distances_train = density.predict(train_embed)
    mind,maxd = min(distances_train),max(distances_train)
    #将正常分数范围保存下来
    with open("logs/%s/minmaxdist.txt"%defect_type,"w") as f_dist:
        f_dist.write("min %.6f max %.6f"%(mind,maxd))
    #计算验证集分数
    distances = density.predict(embeds)
    distances = (distances-mind)/(maxd-mind+1e-8)
    #根据预测分数计算auroc值
    roc_auc = plot_roc(labels, distances, eval_dir / "roc_plot.png", modelname=modelname, save_plots=save_plots)
    return roc_auc


def plot_roc(labels, scores, filename, modelname="", save_plots=False):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # 绘制图片
    if save_plots:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver operating characteristic {modelname}')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(filename)
        plt.close()

    return roc_auc


def plot_tsne(labels, embeds, filename):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500)
    embeds, labels = shuffle(embeds.cpu().detach().numpy(), labels.cpu().detach().numpy())
    tsne_results = tsne.fit_transform(embeds)
    fig, ax = plt.subplots(1)
    colormap = ["b", "r", "c", "y"]

    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], color=[colormap[l] for l in labels])
    fig.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
    parser.add_argument('--model_dir', default="logs",type=str,
                        help=' directory contating models to evaluate (default: logs)')
    parser.add_argument('--data_dir', default="Data",type=str,
                        help=' directory contating datas to evaluate (default: Data)')
    parser.add_argument('--cuda', default=False,
                        help='use cuda for model predictions (default: False)')
    parser.add_argument('--head_layer', default=1, type=int,
                        help='number of layers in the projection head (default: 8)')
    parser.add_argument('--density', default="paddle", choices=["paddle", "sklearn"],
                        help='density implementation to use. See `density.py` for both implementations. (default: paddle)')
    parser.add_argument('--save_plots', default=False,
                        help='save TSNE and roc plots')
    parser.add_argument('--pretrained', default=None,
                        help='no sense')
    args = parser.parse_args()

    all_types = [
                 'bottle',
                 'cable',
                 'capsule',
                 'carpet',
                 'grid',
                 'hazelnut',
                 'leather',
                 'metal_nut',
                 'pill',
                 'screw',
                 'tile',
                 'toothbrush',
                 'transistor',
                 'wood',
                 'zipper'
                 ]
    #根据验证模式决定要验证的类别，全量验证对应15种类别，轻量验证则只验证bottle类别
    if args.type == "all":
        types = all_types
    else:
        types = ["bottle"]
        args.data_dir = "lite_data"
    #决定是否使用gpu
    device = "cuda" if args.cuda in ["True","1","y",True] else "cpu"
    #决定是否输出相关性图片，建议是不要开
    save_plots = True if args.save_plots in ["True","1","y"] else False
    #决定使用的预测分类器，最好使用 GaussianDensityPaddle
    density = GaussianDensitySklearn if args.density == "sklearn" else GaussianDensityPaddle
    print(args)

    # 定义评估结果的保存地址
    eval_dir = Path(args.model_dir + "/evalution")
    eval_dir.mkdir(parents=True, exist_ok=True)
    #计算评估指标并保存为字典形式
    obj = defaultdict(list)
    #遍历每种数据类型进行验证
    for data_type in types:
        print(f"evaluating {data_type}")
        model_name = "%s/%s/final.pdparams"%(args.model_dir,data_type)
        #调用函数计算auroc
        roc_auc = eval_model(model_name, data_type, save_plots=save_plots, device=device,
                             head_layer=args.head_layer, density=density(),data_dir=args.data_dir,args=args)
        #输出并保存评估结果
        print(f"{data_type} AUC: {roc_auc}")
        obj["defect_type"].append(data_type)
        obj["roc_auc"].append(roc_auc)
    ave_auroc = np.mean(obj["roc_auc"])
    obj["defect_type"].append("average")
    obj["roc_auc"].append(ave_auroc)
    print("average auroc:%.4f"%ave_auroc)
    #将评估结果保存为csv文件
    df = pd.DataFrame(obj)
    df.to_csv(str(eval_dir) + "/total_performence.csv")
