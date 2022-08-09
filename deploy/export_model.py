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
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import os
import sys
import numpy as np
paddle.set_device("cpu")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from tools.model import ProjectionNet


def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--model_path', default='logs', help='model_weight')
    parser.add_argument('--headlayer', default=1,type=int, help='headlayer for the model')
    parser.add_argument('--data_type', default='bottle', help='defect type for the model')
    parser.add_argument('--device', default='gpu', help='device')
    parser.add_argument('--img_size', default=256, help='image size to export')
    parser.add_argument(
        '--save-inference-dir', default='deploy', help='path where to save')
    parser.add_argument('--pretrained', default=None, help='pretrained model')
    parser.add_argument('--num_classes', default=3,type=int, help='num_classes')

    args = parser.parse_args()
    return args


def export(args):
    # model = paddlevision.models.__dict__[args.model](
    #     pretrained=args.pretrained, num_classes=args.num_classes)
    head_layers = [512] * args.headlayer + [128]
    model = ProjectionNet(pretrained=args.pretrained, head_layers=head_layers, num_classes=args.num_classes)
    # model = nn.Sequential(model, nn.Softmax())
    model.eval()
    # print("%s/%s/final.pdparmas"%(args.model_path,args.data_type))
    model_dict = paddle.load("%s/%s/final.pdparams"%(args.model_path,args.data_type))
    model.set_dict(model_dict)

    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, args.img_size, args.img_size], dtype='float32')
        ])
    paddle.jit.save(model, os.path.join(args.save_inference_dir, "inference"))
    print(f"inference model has been saved into {args.save_inference_dir}")


if __name__ == "__main__":
    args = get_args()
    export(args)