#-*-coding:utf-8-*-
import os
import shutil
import time
import torch
import json
import time

from tqdm import tqdm
import time

# total参数设置进度条的总长度


class LoggerInfo:
    def __init__(self,config,num_dataset):
        self.loss_all=0
        self.loss_text_all=0
        self.loss_kernel_all=0
        self.step=0
        self.steps=0
        self.t=0
        self.step_all_epoch = int(num_dataset / config.TRAIN.BATCH)
        self.step_epoch=0

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
    def add_step(self):
        self.steps+=1
        self.step_epoch+=1
    def add_time(self,t):
        self.t+=t
    def get_training_time(self):
        t1=self.t
        self.t=0
        return t1
    def clear(self):
        self.step_epoch=0
        self.step=0

def setup_logger(log_file_path: str = None):
    import logging
    from colorlog import ColoredFormatter
    logging.basicConfig(filename=log_file_path, format='%(asctime)s %(levelname)-8s %(filename)s: %(message)s',
                        # 定义输出log的格式
                        datefmt='%Y-%m-%d %H:%M:%S', )
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s %(reset)s %(filename)s: %(message)s",
                                 datefmt='%Y-%m-%d %H:%M:%S',
                                 reset=True,
                                 log_colors={
                                     'DEBUG': 'blue',
                                     'INFO': 'green',
                                     'WARNING': 'yellow',
                                     'ERROR': 'red',
                                     'CRITICAL': 'red',
                                 })

    logger = logging.getLogger('project')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info('logger init finished')
    return logger


def check_outputs(path):
    dirs=os.listdir(path)
    for dir in dirs:

        if len(os.listdir(os.path.join(path,dir)))<3:
            shutil.rmtree(os.path.join(path,dir))


def print_config(config):
    param=''

    param=json.dumps(config,indent=1,separators=(', ',': '),ensure_ascii=False)
    # for key,value in config.items():
    #     param+='{}:{}\n'.format(key,value)
    return param


def train_one_epoch(dataloader,config,psenet,pseloss,optimizer,loggerinfo,logger,schedular,writer,epoch):
    for index, (images, text_kernel_masks, train_masks) in enumerate(dataloader):

        t1 = time.clock()
        # put the batch dataset to GPU
        images = images.to(torch.device('cuda:' + config.CUDA.GPU))
        text_kernel_masks = text_kernel_masks.to(torch.device('cuda:' + config.CUDA.GPU))
        train_masks = train_masks.to(torch.device('cuda:' + config.CUDA.GPU))

        # the model forward
        output = psenet(images,True)

        # calculate the loss
        loss, loss_text, loss_kernel = pseloss(images, output, text_kernel_masks, train_masks)

        # update the weight for the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logger info and tensorboard writer
        loggerinfo.add_step()
        if (index + 1) % config.TRAIN.SHOW_STEP == 0:
            loss_average, loss_text_average, loss_kernel_average = loggerinfo.average_loss()
            logger.info('epoch:[{}/{}],step:[{}/{}] all_step:{}, loss:{:.4f}, loss_s:{:.4f}, loss_k:{:.4f}, {:.2f} samples/s, lr:{}'.
                        format(epoch, config.TRAIN.EPOCH,loggerinfo.step_epoch,loggerinfo.step_all_epoch, loggerinfo.steps, loss_average[0], loss_text_average[0],
                               loss_kernel_average[0],
                               config.TRAIN.BATCH * config.TRAIN.SHOW_STEP / loggerinfo.get_training_time(),
                               schedular.get_lr()[0]))
            writer.add_scalar('Train/Loss', loss_average[0], loggerinfo.steps)
            writer.add_scalar('Train/Loss_s', loss_text_average[0], loggerinfo.steps)
            writer.add_scalar('Train/Loss_kernel', loss_kernel_average[0], loggerinfo.steps)

        else:
            loggerinfo.add_loss(loss.cpu().detach().numpy(), loss_text.cpu().detach().numpy(),
                                loss_kernel.cpu().detach().numpy())
        t2 = time.clock()
        loggerinfo.add_time(t2 - t1)
    loggerinfo.clear()
def valid_one_epoch(dataloader_val,config,psenet,pseloss,optimizer,loggerinfo,logger,schedular,writer,epoch):
    # pbar = tqdm(total=100)
    for index, (images, text_kernel_masks, train_masks) in enumerate(dataloader_val):

        # put the batch dataset to GPU
        images = images.to(torch.device('cuda:' + config.CUDA.GPU))
        text_kernel_masks = text_kernel_masks.to(torch.device('cuda:' + config.CUDA.GPU))
        train_masks = train_masks.to(torch.device('cuda:' + config.CUDA.GPU))

        # the model forward
        output = psenet(images, True)

        # calculate the loss
        loss, loss_text, loss_kernel = pseloss(images, output, text_kernel_masks, train_masks)

        # logger info and tensorboard writer


        # 每次更新进度条的长度
        # time.sleep(0.001)
        # pbar.update(1)


        loggerinfo.add_step()
        loggerinfo.add_loss(loss.cpu().detach().numpy(), loss_text.cpu().detach().numpy(),
                            loss_kernel.cpu().detach().numpy())
    # 关闭占用的资源
    # pbar.close()
    loss_average, loss_text_average, loss_kernel_average = loggerinfo.average_loss()
    logger.info(
        'epoch:[{}/{}],all_step:{}, loss:{:.4f}, loss_s:{:.4f}, loss_k:{:.4f}'.
            format(epoch, config.TRAIN.EPOCH, loggerinfo.step_epoch,
                   loss_average[0], loss_text_average[0],
                   loss_kernel_average[0]))
    writer.add_scalar('Val/Loss', loss_average[0], epoch)
    writer.add_scalar('Val/Loss_s', loss_text_average[0], epoch)
    writer.add_scalar('Val/Loss_kernel', loss_kernel_average[0], epoch)

    #loggerinfo.clear()