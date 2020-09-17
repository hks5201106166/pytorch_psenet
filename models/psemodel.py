#-*-coding:utf-8-*-
from .backbones.resnet import *
from  torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
class PSENET(Module):
    def __init__(self,config):
        super(PSENET, self).__init__()

        self.config=config
        self.backbone=self.get_backbone(config)
        self.layer_conection=self.get_layer_cnnection(self.config)
        self.smooth_layer4 = nn.Sequential(
            nn.Conv2d(config.MODEL.NUM_OUPUT, config.MODEL.NUM_OUPUT, kernel_size=(3,3), padding=1,stride=1),nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True))
        self.smooth_layer3 = nn.Sequential(
            nn.Conv2d(config.MODEL.NUM_OUPUT, config.MODEL.NUM_OUPUT, kernel_size=(3, 3), padding=1,stride=1),nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True))
        self.smooth_layer2 = nn.Sequential(
            nn.Conv2d(config.MODEL.NUM_OUPUT, config.MODEL.NUM_OUPUT, kernel_size=(3, 3), padding=1,stride=1),nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True))
        self.layers_concat_conv=nn.Sequential(nn.Conv2d(config.MODEL.NUM_OUPUT*4,config.MODEL.NUM_OUPUT,kernel_size=(3,3),padding=1,stride=1),
                                              nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True))
        self.layer_output=nn.Sequential(nn.Conv2d(config.MODEL.NUM_OUPUT,config.MODEL.PSE.n,kernel_size=1,stride=1),nn.Sigmoid())

    def forward(self, x,train):
        x2,x3,x4,x5=self.backbone(x)

        x5_layer = self.layer_conection[3](x5)

        x4_connection = self.layer_conection[2](x4)
        x4_layer = self.upsample_add(x5_layer,x4_connection)
        x4_layer = self.smooth_layer4(x4_layer)

        x3_connection = self.layer_conection[1](x3)
        x3_layer = self.upsample_add(x4_layer, x3_connection)
        x3_layer = self.smooth_layer3(x3_layer)

        x2_connection = self.layer_conection[0](x2)
        x2_layer = self.upsample_add(x3_layer, x2_connection)
        x2_layer = self.smooth_layer2(x2_layer)

        layers_concat=self.layers_concat(x2_layer,x3_layer,x4_layer,x5_layer)
        x_last=self.layers_concat_conv(layers_concat)
        output=self.layer_output(x_last)
        if train==True:
            output= F.interpolate(output,size=(x.shape[2:]),mode='bilinear',align_corners=True)

        return output
    def upsample_add(self,input_upsample,input):
        return F.interpolate(input_upsample,size=(input.shape[2:]),mode='bilinear',align_corners=False)+input
    def get_backbone(self,config):
        if config.MODEL.BACKBONE=='resnet18':
            return resnet18(pretrained=config.MODEL.PRETRAINED)
        if config.MODEL.BACKBONE == 'resnet34':
            return resnet34(pretrained=config.MODEL.PRETRAINED)
        if config.MODEL.BACKBONE=='resnet50':
            return resnet50(pretrained=config.MODEL.PRETRAINED)
        if config.MODEL.BACKBONE=='resnet101':
            return resnet101(pretrained=config.MODEL.PRETRAINED)
        if config.MODEL.BACKBONE=='resnet152':
            return resnet152(pretrained=config.MODEL.PRETRAINED)
    def get_layer_cnnection(self,config):
        layer_conection=nn.ModuleList()
        if config.MODEL.BACKBONE=='resnet18':
            for channel in [64,128,256,512]:
                layer_conection.append(nn.Sequential(nn.Conv2d(channel,config.MODEL.NUM_OUPUT,kernel_size=1,stride=1),
                                                     nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True)))
            return layer_conection
        if config.MODEL.BACKBONE == 'resnet34':
            for channel in [64,128,256,512]:
                layer_conection.append(nn.Sequential(nn.Conv2d(channel,config.MODEL.NUM_OUPUT,kernel_size=1,stride=1),
                                                     nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True)))
            return layer_conection
        if config.MODEL.BACKBONE=='resnet50':
            for channel in [256,512,1024,2048]:
                layer_conection.append(nn.Sequential(nn.Conv2d(channel,config.MODEL.NUM_OUPUT,kernel_size=1,stride=1),
                                                     nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True)))
            return layer_conection
        if config.MODEL.BACKBONE=='resnet101':
            for channel in [256,512,1024,2048]:
                layer_conection.append(nn.Sequential(nn.Conv2d(channel,config.MODEL.NUM_OUPUT,kernel_size=1,stride=1),
                                                     nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True)))
            return layer_conection
        if config.MODEL.BACKBONE=='resnet152':
            for channel in [256,512,1024,2048]:
                layer_conection.append(nn.Sequential(nn.Conv2d(channel,config.MODEL.NUM_OUPUT,kernel_size=1,stride=1),
                                                     nn.BatchNorm2d(config.MODEL.NUM_OUPUT),nn.ReLU(inplace=True)))
            return layer_conection
    def layers_concat(self,x2_layer,x3_layer,x4_layer,x5_layer):
        b,c,w,h=x2_layer.shape

        x3_layer = F.interpolate(x3_layer,size=(w,h),mode='bilinear',align_corners=False)
        x4_layer = F.interpolate(x4_layer, size=(w, h), mode='bilinear', align_corners=False)
        x5_layer = F.interpolate(x5_layer, size=(w, h), mode='bilinear', align_corners=False)

        output_layers=torch.cat([x2_layer,x3_layer,x4_layer,x5_layer],dim=1)
        return output_layers
class PSELOSS:
    def __init__(self,config):
        super(PSELOSS, self).__init__()
        self.config=config
    def __call__(self,images,output,text_kernel_masks,train_masks):

        text_masks=text_kernel_masks[:,:,:,-1]
        kernels_masks=text_kernel_masks[:,:,:,:-1]
        text_ps=output[:,-1,:,:]
        kernels_ps=output[:,:-1,:,:]

        loss_text=self.dice_text(images,text_ps,text_masks,train_masks,self.config)

        loss_kernel=self.dice_kernel(text_ps,kernels_ps,kernels_masks,train_masks)
        loss=loss_text*self.config.MODEL.LOSS.LAMBDA+loss_kernel*(1-self.config.MODEL.LOSS.LAMBDA)

        return loss,loss_text,loss_kernel
    def dice_text(self,images,text_ps,text_masks,train_masks,config):
        ratio=config.MODEL.LOSS.OHEM.negation/config.MODEL.LOSS.OHEM.position
        batch_size=text_ps.shape[0]
        ohem_select_masks=[]
        for i in range(batch_size):
            text_p=text_ps[i,:,:]
            text_mask=text_masks[i,:,:]
            train_mask=train_masks[i,:,:]
            image=images[i,:,:,:]
            ohem_select_masks.append(self.ohem(image,text_p,text_mask,train_mask,ratio))
        loss_text=self.dice(text_ps,text_masks,train_masks,ohem_select_masks)
        return loss_text
    def dice_kernel(self,text_ps,kernels_ps,kernels_masks,train_masks):
        batchsize,kernel_nums,_,_=kernels_ps.shape
        kernel_loss_all=torch.Tensor([0.0]).to(torch.device('cuda:'+self.config.CUDA.GPU))
        for i in range(batchsize):
            dice_kernels=torch.Tensor([0.0]).to(torch.device('cuda:'+self.config.CUDA.GPU))
            for n in range(kernel_nums):
                text_p=text_ps[i,:,:].view(-1).float()
                kernels_p=kernels_ps[i,n,:,:].view(-1).float()
                kernels_mask=kernels_masks[i,:,:,n].view(-1).float()

                text_p_mask=text_p>0.5
                text_p_mask=text_p_mask.float()
                dice_kernels+=2*torch.sum((kernels_p*text_p_mask)*(kernels_mask*text_p_mask))/\
                              (torch.sum((kernels_p*text_p_mask)*(kernels_p*text_p_mask))+torch.sum((kernels_mask*text_p_mask)*(kernels_mask*text_p_mask))+0.0001)
            kernel_loss=1-dice_kernels/kernel_nums
            kernel_loss_all+=kernel_loss
        kernel_loss_all/=batchsize
        return kernel_loss_all
    def dice(self,preds,gts,train_masks,select_masks):
        batch_size=preds.shape[0]
        loss=torch.Tensor([0.0]).to(torch.device('cuda:'+self.config.CUDA.GPU))
        for i in range(batch_size):
            pred=preds[i,:,:].view(-1)
            gt=gts[i,:,:].view(-1).float()
            train_mask=train_masks[i].view(-1).float()
            select_mask=select_masks[i].view(-1).float()

            pred_with_mask=pred*train_mask*select_mask
            gt_with_mask=gt*train_mask*select_mask
            loss+=1-(2*torch.sum(pred_with_mask*gt_with_mask))/(torch.sum(pred_with_mask*pred_with_mask)+torch.sum(gt_with_mask*gt_with_mask)+0.0001)
        loss/=batch_size
        return loss
    def ohem(self,image,text_p,text_mask,train_mask,ratio):


        position_nums_train=torch.sum((text_mask==1)&(train_mask==1))
        if position_nums_train.cpu().numpy()==0:
            return train_mask
        negation_nums=torch.sum((text_mask==0)&(train_mask==1))
        if negation_nums.cpu().numpy()==0:
            return train_mask
        negation_nums_train=min(negation_nums,ratio*position_nums_train)

        text_p_sort=text_p[(text_mask==0)&(train_mask==1)].view(-1).sort(descending=True)
        threshold=text_p_sort[0][negation_nums_train.long()-1]

        negation_select_mask=(text_p>threshold)&(text_mask==0)&(train_mask==1)
        # tt=torch.sum((text_p>threshold)&(text_mask==0)&(train_mask==1))
        position_select_mask=(text_mask==1)&(train_mask==1)
        ohem_select_mask=negation_select_mask|position_select_mask

        # text_p=text_p.cpu().detach().numpy()>0.7
        # text_p=np.uint8(text_p)*255
        # cv2.imshow('pred_text',text_p)
        # ohem_select_mask=ohem_select_mask.cpu().numpy()
        # image=np.array(transforms.ToPILImage()(image.cpu()))
        # negation_select_mask=negation_select_mask.cpu().numpy()
        # position_select_mask=position_select_mask.cpu().numpy()
        # cv2.imshow('ttt', image)
        # cv2.imshow('kk',image*np.stack([ohem_select_mask,ohem_select_mask,ohem_select_mask],axis=2))
        # cv2.imshow('neg',image*np.stack([negation_select_mask,negation_select_mask,negation_select_mask],axis=2))
        # cv2.imshow('position', image * np.stack([position_select_mask, position_select_mask, position_select_mask], axis=2))
        # cv2.waitKey(10000)
        return ohem_select_mask





