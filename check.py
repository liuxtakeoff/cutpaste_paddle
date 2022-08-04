"""
读取真实数据集，验证评估指标是否对齐
"""
import random

import cv2
from sklearn.metrics import roc_curve, auc
import sys
import numpy as np
import paddle
import torch
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger
from tools.model import ProjectionNet as net_pp
from model_torch import ProjectionNet as net_torch
# from t.test_dataload_torch import get_loader_torch
# from pp.test_dataload_pp import get_loader_pp
# from t.eval import get_train_embeds as get_train_embeds_torch
# from pp.eval import get_train_embeds as get_train_embeds_pp


import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import KernelDensity
# from pp.density import GaussianDensitySklearn
class GaussianDensitySklearn():
    """Li et al. use sklearn for density estimation.
    This implementation uses sklearn KernelDensity module for fitting and predicting.
    """

    def fit(self, embeddings):
        # estimate KDE parameters
        # use grid search cross-validation to optimize the bandwidth
        self.kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(embeddings)

    def predict(self, embeddings):
        scores = self.kde.score_samples(embeddings)

        # invert scores, so they fit to the class labels for the auc calculation
        scores = -scores

        return scores

# sys.path.append(r"C:\Users\Administrator\Desktop\CUTPASTE\pp")

torch.manual_seed(20227)
paddle.seed(20227)
np.random.seed(20227)
random.seed(20227)

def check_model():
    """
    检查模型并返回加载好权重的两个模型
    """
    print("======start check model...=============")
    # write log
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    data_1 = np.random.rand(3, 3, 256, 256).astype(np.float32)
    # data_2 = np.random.rand(192, 3, 256, 256).astype(np.float32)
    data_2 = data_1
    datap = paddle.to_tensor(data_1)
    datat = torch.tensor(data_2)
    datat = datat.cuda()
    head_layer = 1
    head_layers = [512] * head_layer + [128]
    model_pp = net_pp(pretrained=False, head_layers=head_layers, num_classes=3)
    wt_pp = r"bottle-10000.pdparams"
    model_torch = net_torch(pretrained=False, head_layers=head_layers, num_classes=3)
    model_torch = model_torch.cuda()
    wt_torch = r"bottle-10000.pth"
    model_torch.eval()
    model_pp.eval()
    model_pp.set_dict(paddle.load(wt_pp))
    model_torch.load_state_dict(torch.load(wt_torch))

    # print(model_pp)
    # print(model_torch)
    # wt_pp = paddle.load(wt_pp)
    # wt_torch = torch.load(wt_torch)
    # print(wt_pp.keys(),"\n\n")
    # print(wt_torch.keys())
    #
    # print(wt_pp["out.weight"].shape)
    # print(wt_torch["out.weight"].shape)

    _,datap = model_pp(datap)
    _,datat = model_torch(datat)

    reprod_log_1.add("result_model", datap.cpu().detach().numpy())
    reprod_log_1.save("diff_log/result_model_paddle.npy")

    reprod_log_2.add("result_model", datat.cpu().detach().numpy())
    reprod_log_2.save("diff_log/result_model_torch.npy")

    # check_diff
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("diff_log/result_model_paddle.npy")
    info2 = diff_helper.load_info("diff_log/result_model_torch.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_model.txt")
    return model_pp,model_torch
# def check_loader():
#     """
#     验证数据读取是否准确，并返回两个验证集
#     """
#     print("======start check loader...=============")
#
#     data_type = "bottle" #选择bottle类别来作为验证集
#
#     loader_t,transform_t = get_loader_torch(batch_size=9,data_type=data_type)
#     loader_p,transform_p = get_loader_pp(batch_size=9, data_type=data_type)
#     for batch_idx, (datat,datap) in enumerate(zip(loader_t,loader_p)):
#         datat = datat[0][0].cpu().detach().numpy()
#         datap = datap[0][0].cpu().detach().numpy()
#         imgt = np.transpose(datat,(1,2,0))
#         imgp = np.transpose(datap,(1,2,0))
#         print(imgt[100, 100] - imgp[100,100])
#         # print(imgp[100, 100])
#         # plt.figure(0)
#         # plt.subplot(1,2,1)
#         # plt.title("torch")
#         # plt.imshow(imgt)
#         # plt.subplot(1,2,2)
#         # plt.title("paddle")
#         # plt.imshow(imgp)
#         # plt.show()
#         # plt.close()
#         # print("stop!")
#
#         # print(data[0])
#         # print(len(data))
#         # print(data[0].shape)
#         # print(data[0].size())
#     return loader_p,loader_t,transform_p,transform_t

# def check_eval(loader_pp,loader_torch,model_pp,model_torch,transform_pp,transform_torch):
#     """
#     验证评估指标是否对的上
#     """
#     print("======start check eval...=============")
#     # write log
#     reprod_log_1 = ReprodLogger()
#     reprod_log_2 = ReprodLogger()
#
#     imgsize = 256
#     defect_type = "bottle"
#     device = "cuda"
#     #先训练density
#     density_torch = GaussianDensitySklearn()
#     density_pp = GaussianDensitySklearn()
#     train_embeds_torch=get_train_embeds_torch(model_torch,imgsize,defect_type,transform_torch,device=device,datadir="t/Data")
#     train_embeds_torch = torch.nn.functional.normalize(train_embeds_torch, p=2, dim=1)
#     train_embeds_pp=get_train_embeds_pp(model_pp,imgsize,defect_type,transform_pp,device=device,datadir="t/Data")
#     train_embeds_pp = paddle.nn.functional.normalize(train_embeds_pp, p=2, axis=1)
#     density_pp.fit(train_embeds_pp)
#     density_torch.fit(train_embeds_torch)
#
#     #获取model_torch的输出
#     model_torch.eval()
#     labels_torch = []
#     embeds_torch = []
#     with torch.no_grad():
#         for x, label in loader_torch:
#             x = x.cuda()
#             embed, logit = model_torch(x)
#             # save
#             embeds_torch.append(embed.cpu())
#             labels_torch.append(label.cpu())
#     labels_torch = torch.cat(labels_torch)
#     embeds_torch = torch.cat(embeds_torch)
#     embeds_torch = torch.nn.functional.normalize(embeds_torch, p=2, dim=1)
#     distances = density_torch.predict(embeds_torch)
#     fpr, tpr, _ = roc_curve(labels_torch, distances)
#     roc_auc_torch = auc(fpr, tpr)
#     reprod_log_1.add("result_auroc", np.array([roc_auc_torch]))
#     reprod_log_1.save("diff_log/result_auroc_torch.npy")
#     # 获取model_pp的输出
#     model_pp.eval()
#     labels_pp = []
#     embeds_pp = []
#     with paddle.no_grad():
#         for x, label in loader_pp:
#             # x = x.cuda() #需要转到cuda上
#             embed, logit = model_pp(x)
#             # save
#             embeds_pp.append(embed.cpu())
#             labels_pp.append(label.cpu())
#     labels_pp = paddle.concat(labels_pp)
#     embeds_pp = paddle.concat(embeds_pp)
#     embeds_pp = paddle.nn.functional.normalize(embeds_pp, p=2, axis=1)
#     distances = density_pp.predict(embeds_pp)
#     fpr, tpr, _ = roc_curve(labels_pp, distances)
#     roc_auc_pp = auc(fpr, tpr)
#     reprod_log_2.add("result_auroc", np.array([roc_auc_pp]))
#     reprod_log_2.save("diff_log/result_auroc_paddle.npy")
#     # check_diff
#     diff_helper = ReprodDiffHelper()
#
#     info1 = diff_helper.load_info("diff_log/result_auroc_torch.npy")
#     info2 = diff_helper.load_info("diff_log/result_auroc_paddle.npy")
#
#     diff_helper.compare_info(info1, info2)
#
#     diff_helper.report(
#         diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_auroc.txt")

def check_loss():
    """
    检查损失函数，并返回两个损失函数
    """
    print("======start check loss...=============")
    loss_pp = paddle.nn.CrossEntropyLoss()
    loss_torch = torch.nn.CrossEntropyLoss()
    # write log
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()
    data_1 = np.random.rand(96, 3).astype(np.float32) #随机生成输出数据
    data_2 = np.random.randint(low=0,high=2,size=96).astype(np.int64) #随机生成标签数据
    datap = paddle.to_tensor(data_1,place=paddle.CUDAPlace(0))
    datat = torch.tensor(data_1)
    labelp = paddle.to_tensor(data_2,place=paddle.CUDAPlace(0))
    labelt = torch.tensor(data_2)

    lossp = loss_pp(datap,labelp)
    losst = loss_torch(datat,labelt)

    # reprod_log_1.add("demo_test_1", data_1)
    reprod_log_1.add("result_loss", lossp.cpu().detach().numpy())
    reprod_log_1.save("diff_log/result_loss_paddle.npy")

    # reprod_log_2.add("demo_test_1", data_1)
    reprod_log_2.add("result_loss", losst.cpu().detach().numpy())
    reprod_log_2.save("diff_log/result_loss_torch.npy")

    # check_diff
    diff_helper = ReprodDiffHelper()

    info1 = diff_helper.load_info("diff_log/result_loss_paddle.npy")
    info2 = diff_helper.load_info("diff_log/result_loss_torch.npy")

    diff_helper.compare_info(info1, info2)

    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_loss.txt")
    return loss_pp,loss_torch
    


def check_optim(model_pp,model_torch,test = True):
    """
    检查优化器（学习率是否一致），并返回两个优化器和调度器
    """
    print("======start check optim...=============")
    #定义超参数
    learning_rate = 0.003
    weight_decay = 0.00003 
    momentum = 0.9
    epochs = 100
    #定义优化器及学习率时间表
    optim_torch = torch.optim.SGD(model_torch.parameters(),lr=learning_rate, momentum=momentum,  weight_decay=weight_decay)
    scheduler_torch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim_torch, epochs)

    scheduler_pp = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=learning_rate,T_max=epochs)
    optim_pp = paddle.optimizer.Momentum(parameters=model_pp.parameters(),learning_rate=scheduler_pp, momentum=momentum,  weight_decay=weight_decay)
    
    if test:
        # write log
        reprod_log_1 = ReprodLogger()
        reprod_log_2 = ReprodLogger()
        lr_pps = []
        lr_torchs = []
        for step in range(epochs-1):
            scheduler_torch.step()
            scheduler_pp.step()
            lr_pp = optim_pp.get_lr()
            lr_pps.append(lr_pp)
            lr_torch = optim_torch.param_groups[0]["lr"]
            lr_torchs.append(lr_torch)
        lr_pps = np.array(lr_pps)
        lr_torchs = np.array(lr_torchs)
        # print("lr_pp:\n",lr_pps)
        # print("lr_torch:\n",lr_torchs)
        reprod_log_1.add("result_lr", lr_pps)
        reprod_log_1.save("diff_log/result_lr_paddle.npy")

        reprod_log_2.add("result_lr", lr_torchs)
        reprod_log_2.save("diff_log/result_lr_torch.npy")

        # check_diff
        diff_helper = ReprodDiffHelper()

        info1 = diff_helper.load_info("diff_log/result_lr_paddle.npy")
        info2 = diff_helper.load_info("diff_log/result_lr_torch.npy")

        diff_helper.compare_info(info1, info2)

        diff_helper.report(
            diff_method="mean", diff_threshold=1e-6, path="diff_log/diff_LearningRate.txt")
    else:
        return optim_pp,optim_torch,scheduler_pp,scheduler_torch


def check_backward(model_pp,model_torch,optim_pp,optim_torch,scheduler_pp,scheduler_torch,loss_pp,loss_torch):
    """
    检查反向传播过程
    """
    print("======start check backward...=============")
    # write log
    reprod_log_1 = ReprodLogger()
    reprod_log_2 = ReprodLogger()

    data_1 = np.random.rand( 9,3, 256, 256).astype(np.float32) #随机生成训练数据
    datap = paddle.to_tensor(data_1)
    yp = paddle.arange(3)
    yp = paddle.repeat_interleave(yp, 3,0)
    datat = torch.tensor(data_1)
    yt = torch.arange(3,device="cuda")
    yt = yt.repeat_interleave(3)
    datat = datat.cuda()

    #开始反向传播
    model_pp.eval()
    model_torch.eval()
    loss_list_pp = []
    lr_list_pp = []
    lr_list_torch = []
    loss_list_torch = []
    for i in range(3):
        #清零梯度
        optim_torch.zero_grad()
        optim_pp.clear_grad()
        #前向推理并获得损失
        embeds_pp,logits_pp = model_pp(datap)
        embeds_torch,logits_torch = model_torch(datat)
        lossp = loss_pp(logits_pp,yp)
        losst = loss_torch(logits_torch,yt)
        with torch.no_grad():
            with paddle.no_grad():
                #保存权重
                loss_list_pp.append(lossp.cpu().detach().numpy())
                lr_list_pp.append(np.array(optim_pp.get_lr()))
                loss_list_torch.append(losst.cpu().detach().numpy())
                lr_list_torch.append(np.array(optim_torch.param_groups[0]["lr"]))
        #迭代优化
        losst.backward()
        # for name, tensor in model_torch.named_parameters():
        #     grad = tensor.grad
        #     print(name, tensor.grad)
        optim_torch.step()
        scheduler_torch.step()
        lossp.backward()
        # for name, tensor in model_pp.named_parameters():
        #     grad = tensor.grad
        #     print(name, tensor.grad)
        optim_pp.step()
        scheduler_pp.step()
    
    #逐次检查损失是否对齐
    diff_helper = ReprodDiffHelper()
    for i in range(3):
        #检查损失值
        reprod_log_1.add("loss", loss_list_pp[i])
        reprod_log_1.save("diff_log/loss_epoch%d_paddle.npy"%i)

        reprod_log_2.add("loss", loss_list_torch[i])
        reprod_log_2.save("diff_log/loss_epoch%d_torch.npy"%i)

        info1 = diff_helper.load_info("diff_log/loss_epoch%d_paddle.npy"%i)
        info2 = diff_helper.load_info("diff_log/loss_epoch%d_torch.npy"%i)

        diff_helper.compare_info(info1, info2)
        #检查学习率
        reprod_log_1.add("lr", lr_list_pp[i])
        reprod_log_1.save("diff_log/lr_epoch%d_paddle.npy" % i)

        reprod_log_2.add("lr", lr_list_torch[i])
        reprod_log_2.save("diff_log/lr_epoch%d_torch.npy" % i)

        info1 = diff_helper.load_info("diff_log/lr_epoch%d_paddle.npy" % i)
        info2 = diff_helper.load_info("diff_log/lr_epoch%d_torch.npy" % i)

        diff_helper.compare_info(info1, info2)

        diff_helper.report(
            diff_method="mean", diff_threshold=1e-2, path="diff_log/diff_backward_epoch%d.txt"%i)

        







if __name__ == '__main__':
    model_pp,model_torch = check_model()
    # loader_pp, loader_torch,transform_pp,transform_torch = check_loader()
    # check_eval(loader_pp, loader_torch,model_pp,model_torch,transform_pp,transform_torch)
    loss_pp,loss_torch = check_loss()
    check_optim(model_pp, model_torch,test=True)
    optim_pp,optim_torch,scheduler_pp,scheduler_torch = check_optim(model_pp, model_torch,test=False)
    model_torch.freeze_resnet()
    model_pp.freeze_resnet()
    check_backward(model_pp, model_torch, optim_pp, optim_torch, scheduler_pp, scheduler_torch,loss_pp,loss_torch)
    pass