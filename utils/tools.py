#-*-coding:utf-8-*-
class AverageLoss:
    def __init__(self):
        self.loss_all=0
        self.loss_text_all=0
        self.loss_kernel_all=0
        self.step=0
    def add_loss(self,loss,loss_text,loss_kernel):
        self.loss_all+=loss
        self.loss_text_all+=loss_text
        self.loss_kernel_all+=loss_kernel
        self.step+=1
    def average_loss(self):
        loss=self.loss_all/self.step
        loss_text=self.loss_text_all/self.step
        loss_kernel=self.loss_kernel_all/self.step
        self.loss_all = 0
        self.loss_text_all = 0
        self.loss_kernel_all = 0
        self.step = 0
        return loss,loss_text,loss_kernel
