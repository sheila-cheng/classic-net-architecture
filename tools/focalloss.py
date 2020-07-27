"""
  combination of "cross entropy loss(weighted)" and "modulating factor"

  hyperparameter:
    weight(alpha): balance multi-classes frequency
        ::alpha:: scalar weight for bi-classification positive-class)
        ::weight:: one dimension tensor weight for multi-class
    gamma: encourge the optimizing process to pay less attension to easy examples

  ps：understanding the mechanism of CrossEntropy_loss, i.g the relationship of CrossEntropy_loss, log_sofmax and nll_loss
"""

import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss,self).__init__()

        self.weight = weight
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

        if len(self.weight) == 1:
            self.mode = "binary"
        else:
            self.mode = "multi"

    def forward(self, outs, labels):
        """
        suitable for classificaion/segmentation task of bi/multi classes

        ::outs:: size of (batchsize)/(batchsize, num_classes)/(batchsie, num_classes, H, W)
        ::labels:: size of (batchsize)/(batchsize, H, W)
        """

        labels.to(torch.LongTensor) # torch.int64

        # ::modulating factor:: related to every sample's output probability (tensor of size whole-batch-data)
        # ::traditional weight:: related to every class's frequency (vector of length num_classes)

        if self.mode == "binary":
            loss_fn = torch.nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=self.weight)
            focal_loss = loss_fn(outs, labels)
        else:
            p = F.softmax(outs, dim=1)
            modulating_factor = (1 - p) ** self.gamma

            loss_fn = torch.nn.NLLLoss(weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
            focal_loss = loss_fn(modulating_factor*p, labels)

        return focal_loss
