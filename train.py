#-*-coding:utf-8-*-
from easydict import EasyDict as edict
import yaml
import argparse
import torch
from datasets.datasets import Dataset_PSE
from torch.utils.data import DataLoader
from models.psemodel import PSENET,PSELOSS
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
i=0
for images,text_kernel_masks,train_masks in dataloader:
    i+=1
    images=images.to(torch.device('cuda:'+config.CUDA.GPU))
    text_kernel_masks=text_kernel_masks.to(torch.device('cuda:'+config.CUDA.GPU))
    train_masks=train_masks.to(torch.device('cuda:'+config.CUDA.GPU))
    with torch.no_grad():
        output = psenet(images)
