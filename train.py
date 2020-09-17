#-*-coding:utf-8-*-
from easydict import EasyDict as edict
import yaml
import argparse
import torch
from torch.autograd import Variable
from datasets.datasets import Dataset_PSE
from torch.utils.data import DataLoader
from models.psemodel import PSENET,PSELOSS
from utils.lr_scheduler import get_scheduler
from utils.tools import *
import os
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import torchvision

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

if __name__ == "__main__":
    #define the config
    config=config_args()

    #define the train dataset
    dataset=Dataset_PSE(config=config)
    dataloader = DataLoader(dataset=dataset, batch_size=config.TRAIN.BATCH, shuffle=config.TRAIN.SHUFFLE, num_workers=config.TRAIN.WORKERS)



    #define the psenet model , psenet loss , optimizer and schedular
    psenet=PSENET(config=config).to(torch.device('cuda:'+config.CUDA.GPU))
    pseloss=PSELOSS(config=config)
    optimizer=torch.optim.Adam(psenet.parameters(),lr=config.TRAIN.LR)

    schedular=get_scheduler(config,optimizer)

    #define the logger and the tensorboard writer

    loggerinfo=LoggerInfo(config,num_dataset=len(dataset))
    if os.path.exists(config.MODEL.MODEL_SAVE_DIR)==False:
        os.mkdir(config.MODEL.MODEL_SAVE_DIR)
    else:
        check_outputs(config.MODEL.MODEL_SAVE_DIR)
    nowtime=datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')
    os.mkdir(os.path.join(config.MODEL.MODEL_SAVE_DIR,nowtime))
    os.mkdir(os.path.join(config.MODEL.MODEL_SAVE_DIR+'/'+nowtime,'runs'))
    logger=setup_logger(log_file_path=os.path.join(config.MODEL.MODEL_SAVE_DIR+'/'+nowtime,'log.txt'))
    logger.info(print_config(config))
    writer=SummaryWriter(config.MODEL.MODEL_SAVE_DIR+'/'+nowtime+'/runs')


    #model resume
    start_epoch=0
    if config.TRAIN.RESUME.FLAG==True:
        logger.info('model resume:'+config.TRAIN.RESUME.MODEL_SAVE_PATH)
        checkpoint = torch.load(config.TRAIN.RESUME.MODEL_SAVE_PATH)

        psenet.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    for epoch in range(start_epoch,config.TRAIN.EPOCH):
        schedular.step(epoch=epoch)
        #for one training epoch
        psenet.train()
        loss, loss_tex, loss_kernel=train_one_epoch(dataloader, config, psenet, pseloss, optimizer, loggerinfo, logger, schedular, writer, epoch)
        # for one validation epoch
        if (0.3<loss<0.4 and epoch%4==0) or (epoch+1)%10==0 or loss<0.2:
            psenet.eval()
            recall,precision,f1=valid_one_epoch(config,psenet,pseloss,optimizer,schedular,writer,epoch)
            state = {'net':psenet.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            torch.save(state,config.MODEL.MODEL_SAVE_DIR+'/'+nowtime+'/'+str(epoch)+'_f1:'+str(f1)+'_'+'recall:'+str(recall)+'_'+'precision:'+str(precision)+'_model.pth')
            logger.info(str(epoch)+'f1:'+str(f1)+'_'+'recall:'+str(recall)+'_'+'precision:'+str(precision))


    writer.close()
