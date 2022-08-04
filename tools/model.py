import paddle.nn as nn
import paddle

class head(nn.Layer):
    def __init__(self, pretrained=False, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
        super(head, self).__init__()


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


class ProjectionNet(nn.Layer):
    def __init__(self, pretrained=False, head_layers=[512, 512, 512, 512, 512, 512, 512, 512, 128], num_classes=3):
        super(ProjectionNet, self).__init__()
        self.resnet18 = paddle.vision.models.resnet18(pretrained=False)
        if pretrained:
            self.resnet18.load_dict(paddle.load("resnet18_pretrianed_paddle.pdparams"))
        last_layer = 512
        sequential_layers = []
        for num_neurons in head_layers:
            sequential_layers.append(nn.Linear(last_layer, num_neurons))
            sequential_layers.append(BatchNorm1D_new(num_neurons,affine=True,track_running_stats=True))
            sequential_layers.append(nn.ReLU())
            last_layer = num_neurons
        head = nn.Sequential(
            *sequential_layers
        )
        self.resnet18.fc = nn.Identity()
        self.head = head
        self.out = nn.Linear(last_layer, num_classes, bias_attr=True)

    def forward(self, x):
        embeds = self.resnet18(x)
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
