import torch.nn as nn
import torch

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.iters = 0

    @staticmethod
    def abs_exp_loss(y_pred, y_true, pow):
        return torch.abs((y_pred - y_true) ** pow).mean()

    def forward(self, y_pred, y_true):
        dscs = torch.zeros(y_pred.shape[1])

        for i in range(y_pred.shape[1]):
          y_pred_ch = y_pred[:, i].contiguous().view(-1)
          y_true_ch = y_true[:, i].contiguous().view(-1)
          intersection = (y_pred_ch * y_true_ch).sum()
          dscs[i] = (2. * intersection + self.smooth) / (
              y_pred_ch.sum() + y_true_ch.sum() + self.smooth
          )

        return (1. - torch.mean(dscs))