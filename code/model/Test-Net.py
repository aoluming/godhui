import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.resnet import *


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.3):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        nn.init.normal(self.linear_1.weight, std=0.001)
        nn.init.normal(self.linear_2.weight, std=0.001)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# standard NORM layer of Transformer
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6, trainable=True):
        super(Norm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        if trainable:
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))
        else:
            self.alpha = nn.Parameter(torch.ones(self.size), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(self.size), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class Transform(nn.Module):
    def __init__(self, d_model=64, dropout=0.3):
        super(Transform, self).__init__()
        self.d_model = d_model
        # no of head has been modified to encompass : 1024 dimension
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff=d_model / 2)

    def forward(self, q, k, v, mask=None):
        # q: (b , dim )
        b = q.size(0)
        t = k.size(1)
        dim = q.size(1)
        q_temp = q.unsqueeze(1)
        q_temp = q_temp.expand(b, t, dim)
        # q,k,v : (b, t , d_model=1024 // 16 )
        A = self.attention(q_temp, k, v, self.d_model, mask, self.dropout)
        # A : (b , d_model=1024 // 16 )
        q_ = self.norm_1(A + q)
        new_query = self.norm_2(q_ + self.dropout_2(self.ff(q_)))
        return new_query

    def attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.sum(q * k, -1) / math.sqrt(d_k)
        # scores : b, t
        scores = F.softmax(scores, dim=-1)
        scores = scores.unsqueeze(-1).expand(scores.size(0), scores.size(1), v.size(-1))
        # scores : b, t, dim
        output = scores * v
        output = torch.sum(output, 1)
        if dropout:
            output = dropout(output)
        return output

class Block_head(nn.Module):
    def __init__(self, d_model=64, dropout=0.3):
        super(Block_head, self).__init__()
        self.T1 = Transform()
        self.T2 = Transform()
        self.T3 = Transform()

    def forward(self, q, k, v, mask=None):
        q = self.T1(q, k, v)
        q = self.T2(q, k, v)
        q = self.T3(q, k, v)
        return q


class CEN(nn.Module):

    def __init__(self):
        super(CEN, self).__init__()
        # 其实这里可以这样写
        # self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        self.backbone_img = nn.Sequential(
            resnet18(pretrained=True),
            nn.AdaptiveMaxPool2d(14),
            nn.BatchNorm2d(512)
        )
        self.backbone_face = nn.Sequential(
            resnet18(pretrained=True),
            nn.AdaptiveMaxPool2d(1),
            nn.InstanceNorm2d(512)
        )
        self.attention = Transform()
        # downsample 16 times
        # self.transformer = Transform()

    def forward(self, image, face):
        fm_img = self.backbone_img(image)
        fm_face = self.backbone_face(face)
        fm_face = fm_face.view(fm_face.size(0), -1)

        return fm_img, fm_face


if __name__ == '__main__':
    img = torch.rand(size=(1, 3, 480, 480))
    face = torch.rand(size=(1, 3, 64, 64))
    # downsample 16 times
    net = CEN()
    print(net(img, face)[1].size())
