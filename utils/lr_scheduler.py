#-*-coding:utf-8-*-



"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from bisect import bisect_right
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torchvision
import matplotlib.pyplot as plt
# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1/3,
        warmup_iters=100,
        warmup_method="linear",
        last_epoch=-1,):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def get_scheduler(config,optimizer):
    if config.TRAIN.SCHEDULER.IS_MultiStepLR:
        return lr_scheduler.MultiStepLR(optimizer,milestones=config.TRAIN.SCHEDULER.MultiStepLR.LR_STEP,gamma=config.TRAIN.SCHEDULER.MultiStepLR.LR_FACTOR,last_epoch=-1)
    else:
        return  WarmupMultiStepLR(optimizer=optimizer,
                              milestones=config.TRAIN.SCHEDULER.WarnUpLR.milestones,
                              gamma=config.TRAIN.SCHEDULER.WarnUpLR.gamma,
                              warmup_factor=config.TRAIN.SCHEDULER.WarnUpLR.warmup_factor,
                              warmup_iters=config.TRAIN.SCHEDULER.WarnUpLR.warmup_iters,
                              warmup_method=config.TRAIN.SCHEDULER.WarnUpLR.warmup_method,
                              last_epoch=-1)



if __name__ == "__main__":
    optimizer = torch.optim.Adam(torchvision.models.resnet18(pretrained=False).parameters(), lr=0.01)
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                  milestones=[20,80],
                                  gamma=0.1,
                                  warmup_factor=0.1,
                                  warmup_iters=2,
                                  warmup_method="linear",
                                  last_epoch=-1)
    # For updating learning rate
    def update_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    x=[]
    y=[]
    for epoch in range(120):
      scheduler.step()
      y.append(scheduler.get_lr())
      x.append(epoch)
    print(y)
    plt.plot(x,y)
    plt.show()

