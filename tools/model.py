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
import paddle.nn as nn
import paddle

class head(nn.Layer):
    def __init__(self, pretrained=False, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
        super(head, self).__init__()

#自定义带affine和track_running_stats参数的batch_normal层
class BatchNorm1D_new(paddle.nn.BatchNorm1D):
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 momentum=0.1,
                 affine=True,
                 track_running_stats=True):
        momentum = 1 - momentum
        weight_attr = None
        bias_attr = None
        if not affine:
            weight_attr = paddle.ParamAttr(learning_rate=0.0)
            bias_attr = paddle.ParamAttr(learning_rate=0.0)
        super().__init__(
            num_features,
            momentum=momentum,
            epsilon=eps,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=track_running_stats)

#定义模型
class ProjectionNet(nn.Layer):
    def __init__(self, pretrained=False, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
        super(ProjectionNet, self).__init__()
        #使用paddle自带的resnet18模型
        self.resnet18 = paddle.vision.models.resnet18(pretrained=False)
        #如果需要，加载resnet18预训练权重，该权重需要自行下载，百度网盘地址：https://pan.baidu.com/s/1QJkda31WcaY9ngALvWsGDw，提取码：l7c3
        if pretrained:
            self.resnet18.load_dict(paddle.load("resnet18_pretrianed_paddle.pdparams"))
        last_layer = 512
        sequential_layers = []
        #根据head_layers层数，搭建head模块
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(BatchNorm1D_new(num_neurons,affine=True,track_running_stats=True))
            sequential_layers.append(nn.ReLU())
            last_layer = num_neurons
        head = nn.Sequential(
            *sequential_layers
        )
        #去除resnet线性分类层
        self.resnet18.fc = nn.Identity()
        self.head = head
        #搭建输出模块
        self.out = nn.Linear(last_layer, num_classes, bias_attr=True)

    def forward(self, x):
        #原始输入->深度特征
        embeds = self.resnet18(x)
        #深度特征->预测结果
        tmp = self.head(embeds)
        logits = self.out(tmp)
        return embeds, logits

    def freeze_resnet(self):
        # freez full resnet18
        for param in self.resnet18.parameters():
            param.stop_gradient = True

        # unfreeze head:
        for param in self.resnet18.fc.parameters():
            param.stop_gradient = False

    def unfreeze(self):
        # unfreeze all:
        for param in self.parameters():
            param.stop_gradient = False
