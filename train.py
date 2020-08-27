#-*-coding:utf-8-*-
from easydict import EasyDict as edict
import yaml
import argparse
from datasets.datasets import Dataset_PSE
from torch.utils.data import DataLoader
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
dataloader=DataLoader(dataset=dataset,batch_size=8,shuffle=True,num_workers=0)
for images,text_kernel_masks,train_masks in dataloader:
    pass