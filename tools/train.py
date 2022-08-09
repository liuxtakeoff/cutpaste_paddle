# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import time
import paddle
from paddle import optimizer as optim
from paddle.io import DataLoader
from paddle.optimizer.lr import CosineAnnealingDecay
from paddle.vision import transforms
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))
from tools.dataset import MVTecAT, Repeat
from tools.cutpaste import CutPasteNormal, CutPasteScar, CutPaste3Way, CutPasteUnion, cut_paste_collate_fn
from tools.model import ProjectionNet
from tools.eval import eval_model
import os
import random
import numpy as np
seed = 2022952
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

def run_training(data_type="bottle",
                 model_dir="logs",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=8,
                 cutpate_type=CutPasteNormal,
                 device="cuda",
                 workers=8,
                 size=256,
                 save_interval=1000,
                 args = None):
    #创建输出文件夹
    if not os.path.exists("%s/%s"%(model_dir,data_type)):
        os.makedirs("%s/%s"%(model_dir,data_type))
    weight_decay = args.weight_decay  #定义优化器权重衰减系数
    momentum = args.momentum          #定义优化器动量值
    # augmentation:
    min_scale = args.min_scale        #定义缩放变换尺度，默认为1，不进行变换
    hsv_jitter = args.hsv_jitter      #定义color_jitter变化尺度，默认为0.1
    # 创建图像预处理方法
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))
    #创建图像增强方法
    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.RandomResizedCrop(size, scale=(min_scale,1)))
    train_transform.transforms.append(transforms.ColorJitter(brightness=hsv_jitter, contrast=hsv_jitter, saturation=hsv_jitter, hue=hsv_jitter))
    train_transform.transforms.append(transforms.Resize((size, size)))
    train_transform.transforms.append(cutpate_type(transform=after_cutpaste_transform))

    #创建训练数据集
    train_data = MVTecAT(args.data_dir, data_type, transform=train_transform, size=int(size * (1 / min_scale)))
    #创建训练数据加载器
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=cut_paste_collate_fn,
                            persistent_workers=True, prefetch_factor=5)

    # 创建模型
    head_layers = [512] * head_layer + [128]
    num_classes = 2 if cutpate_type is not CutPaste3Way else 3
    model = ProjectionNet(pretrained=pretrained, head_layers=head_layers, num_classes=num_classes)
    #如果设置了读取预训练权重及初始冻结，则冻结主干模型权重
    if freeze_resnet > 0 and pretrained:
        model.freeze_resnet()
    model.train()
    #定义损失函数
    loss_fn = paddle.nn.CrossEntropyLoss()
    #根据设置选择优化器
    if optim_name == "sgd":
        scheduler = CosineAnnealingDecay(learning_rate=learninig_rate, T_max=epochs)
        optimizer = optim.Momentum(parameters=model.parameters(), learning_rate=scheduler, momentum=momentum,
                                   weight_decay=weight_decay)
    elif optim_name == "adam":
        optimizer = optim.Adam(parameters=model.parameters(), learning_rate=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")
    #定义数据加载函数，实现持续加载
    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out
    dataloader_inf = get_data_inf()

    total_loss = 0          #初始化总损失值
    total_reader_cost = 0   #初始化总数据读取时间
    t0 = time.time()        #记录当前时间，用于计算训练总用时
    max_auroc = 0           #初始化最佳auroc值
    #定义输出日志文件
    fdata = open("%s/%s/total_epochs.csv"%(model_dir,data_type),"w")
    fdata.write("epoch,loss,auroc\n")
    #开始定步数训练，默认步数为10000
    for epoch in range(epochs):
        #early_stop
        if max_auroc>=0.9995:break
        #到达解锁步数，则解锁主干网络权重
        if epoch == freeze_resnet:
            model.unfreeze()

        t1 = time.time() #记录当前时刻，用以计算数据读取时间
        batch_idx, data = next(dataloader_inf) #读取一个批次数据
        #堆叠img，img-cutpaste,img-cutpaste(scar)用以匹配模型输入
        xs = [x for x in data]
        xc = paddle.concat(xs, axis=0)
        # 清零优化器梯度
        optimizer.clear_grad()

        total_reader_cost += time.time() - t1 #更新数据读取时间
        #前向推理
        embeds, logits = model(xc)

        # 计算理论标签值，img对应0，img-cutpaste对应1，img-cutpaste(scar)对应2
        y = paddle.arange(len(xs))
        y = paddle.repeat_interleave(y, xs[0].shape[0], 0)
        #计算损失值
        loss = loss_fn(logits, y)
        # 反向传播，更新参数
        loss.backward()
        optimizer.step()
        #更新学习率调度器
        if scheduler is not None:
            scheduler.step()
        with paddle.no_grad():
            #计算这一批次训练的正确率
            predicted = paddle.argmax(logits, axis=1)
            accuracy = paddle.divide(paddle.sum(predicted == y), paddle.to_tensor(predicted.shape[0]))
            #如果步数间隔了一个test_epochs，进行一次模型验证
            if test_epochs>0 and (epoch+1)%test_epochs == 0 and epoch>0:
                batch_embeds = []
                batch_embeds.append(embeds.cpu().detach())
                model.eval()
                #计算模型auroc值
                roc_auc = eval_model("", data_type, device=device,
                                     save_plots=False,
                                     size=size,
                                     show_training_data=False,
                                     model=model,
                                     args = args)
                #保存验证值
                fdata.write("%d,%f,%f\n"%(epoch+1,loss,roc_auc))
                # print("epoch:%d type:%s auroc:%f(%f)"%(epoch,data_type,roc_auc,max_auroc))
                #更新最新权重
                if roc_auc >= max_auroc:
                    max_auroc = roc_auc
                    paddle.save(model.state_dict(), os.path.join(str(model_dir),data_type,
                                                             "final.pdparams"))
                model.train()
            #如果步数间隔了一个log_interval，则输出一次日志
            if (epoch+1) % args.log_interval == 0:
                total_bacth_cost = time.time() - t0

                print("epoch:%d/%d loss:%.4f acc:%.3f avg_reader_cost:%.3f avg_batch_cost:%.3f avg_ips:%.3f lr:%.6f"%(
                epoch+1,epochs,loss,accuracy,total_reader_cost/(epoch+1),total_bacth_cost/(epoch+1),total_bacth_cost/(epoch+1)/batch_size,optimizer.get_lr()
                ))
    fdata.close()
    #训练结束了，保存最终模型
    if test_epochs<0 or test_epochs >= epochs:
        paddle.save(model.state_dict(), os.path.join(str(model_dir),data_type,"final.pdparams"))


if __name__ == '__main__':
    # place = paddle.CUDAPlace(0)
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all",help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')
    parser.add_argument('--epochs', default=256, type=int,help='number of epochs to train the model , (default: 256)')
    parser.add_argument('--model_dir', default="logs",help='output folder of the models , (default: models)')
    parser.add_argument('--data_dir', default="Data",help='path of data , (default: Data)')
    parser.add_argument('--pretrained', default=False, type=bool,help='use pretrained values to initalize ResNet18 , (default: False)')
    parser.add_argument('--test_epochs', default=50, type=int,help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')
    parser.add_argument('--freeze_resnet', default=20, type=int,help='number of epochs to freeze resnet (default: 20)')
    parser.add_argument('--lr', default=0.03, type=float,help='learning rate (default: 0.03)')
    parser.add_argument('--optim', default="sgd",help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')
    parser.add_argument('--batch_size', default=32, type=int,help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')
    parser.add_argument('--head_layer', default=1, type=int,help='number of layers in the projection head (default: 1)')
    parser.add_argument('--variant', default="3way", choices=['normal', 'scar', '3way', 'union'],help='cutpaste variant to use (dafault: "3way")')
    parser.add_argument('--cuda', default=True,help='use cuda for training (default: False)')
    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:8)")
    parser.add_argument('--save_interval', default=1000, type=int, help="number of epochs between each model save (default:1000)")
    parser.add_argument('--log_interval', default=1, type=int, help="number of step between each log print (default:1)")
    parser.add_argument('--weight_decay', default=0.00003, type=float, help="weight decay in optimzer (default:0.00003)")
    parser.add_argument('--momentum', default=0.9, type=float, help="momentum in optimzer (default:0.9)")
    parser.add_argument('--min_scale', default=1, type=float, help="min_scale param in data augmentation (default:1)")
    parser.add_argument('--hsv_jitter', default=0.1, type=float, help="hsv_jitter scale in colorjitter (default:0.1)")
    parser.add_argument('--density', default="paddle", choices=["paddle", "sklearn"],
                        help='density implementation to use. See `density.py` for both implementations. (default: paddle)')
    parser.add_argument('--output', default=None, help="no sense")
    args = parser.parse_args()
    #定义全量训练数据类型，共计15个数据类型
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
    #根据训练模式决定要训练的类别，全量训练对应15种类别，轻量训练则只训练bottle类别
    if args.type == "all":
        types = all_types
    else:
        types = ["bottle"]
        args.data_dir = "lite_data"
    print(args)
    #根据参数选择cutpaste类型，默认为3way类型
    variant_map = {'normal': CutPasteNormal, 'scar': CutPasteScar, '3way': CutPaste3Way, 'union': CutPasteUnion}
    variant = variant_map[args.variant]
    #设置训练环境
    device = "gpu" if args.cuda in ["True","1","y",True] else "cpu"
    print(f"using device: {device}")
    paddle.set_device(device)
    # 创建输出文件夹
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # 保存训练参数
    with open(Path(args.model_dir) / "run_config.txt", "a") as f:
        f.write(str(args)+"\n")
    #遍历每种数据类型进行训练
    for data_type in types:
        print(f"training {data_type}")
        run_training(data_type,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_resnet=args.freeze_resnet,
                     learninig_rate=args.lr,
                     optim_name=args.optim,
                     batch_size=args.batch_size,
                     head_layer=args.head_layer,
                     device=device,
                     cutpate_type=variant,
                     workers=args.workers,
                     save_interval=args.save_interval,
                     args=args)
