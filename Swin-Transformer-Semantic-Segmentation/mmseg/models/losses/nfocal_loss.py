import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class NFocalLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 gamma=2,
                 loss_weight=1.0,
                 ignore_index=255,
                 loss_name='nfoclal_loss'):
        super(NFocalLoss, self).__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self._loss_name = loss_name

    def forward(self, logit, target,  **kwargs):
        # n, c, h, w = logit.size()
        criterion = self.ce
        logpt = -criterion(logit, target)
        pt = torch.exp(logpt)
        fl = -((1 - pt) ** self.gamma) * logpt
        z = criterion(logit, target).sum()/fl.sum()
        z = z.detach()
        # print(z, z.grad_fn)
        loss = z * ((1-pt)**self.gamma) * criterion(logit, target)
        loss = loss.mean()
        return self.loss_weight * loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name