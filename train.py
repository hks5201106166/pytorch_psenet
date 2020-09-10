#-*-coding:utf-8-*-
from easydict import EasyDict as edict
import yaml
import argparse
import torch
from datasets.datasets import Dataset_PSE
from torch.utils.data import DataLoader
from models.psemodel import PSENET,PSELOSS
from utils.lr_scheduler import get_scheduler
from utils.tools import AverageLoss
def config_args():
    '''
    define the config
    '''
    parser = argparse.ArgumentParser()
    # the config for the train
    parser.add_argument('--config',default='config/config.yaml',type=str,help='the path of the images dir')
    args = parser.parse_args()
    with open(args.config,'r') as file:
        config=yaml.load(file)
        config=edict(config)
    return config
config=config_args()
dataset=Dataset_PSE(config=config)
psenet=PSENET(config=config,train=True).to(torch.device('cuda:'+config.CUDA.GPU))
dataloader=DataLoader(dataset=dataset,batch_size=config.TRAIN.BATCH,shuffle=True,num_workers=0)
pseloss=PSELOSS(config=config)
optimizer=torch.optim.Adam(psenet.parameters(),lr=config.TRAIN.LR)
schedular=get_scheduler(config,optimizer)
averager=AverageLoss()

for epoch in range(1000):
    schedular.step()

    for index,(images,text_kernel_masks,train_masks) in enumerate(dataloader):

        images=images.to(torch.device('cuda:'+config.CUDA.GPU))
        text_kernel_masks=text_kernel_masks.to(torch.device('cuda:'+config.CUDA.GPU))
        train_masks=train_masks.to(torch.device('cuda:'+config.CUDA.GPU))

        output = psenet(images)
        loss,loss_text,loss_kernel=pseloss(images,output,text_kernel_masks,train_masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # schedular.step()
        if (index+1)%config.TRAIN.SHOW_STEP==0:
            loss_average, loss_text_average, loss_kernel_average=averager.average_loss()
            print('epoch:{},loss is {},loss_s is {},loss_k is {}'.format(epoch,loss_average, loss_text_average, loss_kernel_average))
        else:
            averager.add_loss(loss.cpu().detach().numpy(),loss_text.cpu().detach().numpy(),loss_kernel.cpu().detach().numpy())
