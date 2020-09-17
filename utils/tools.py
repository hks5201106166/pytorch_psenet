#-*-coding:utf-8-*-
import os
import shutil
import time
import torch
import json
import time
import numpy as np
from tqdm import tqdm
import time
import cv2
import tqdm
import torchvision
from pse import decode as pse_decode
from cal_recall import cal_recall_precison_f1
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
def load_imagename_labels(labels_root):
    labels=os.listdir(labels_root)
    labels_dict={}
    image_names=[]
    for label_file in labels:
        boxs=[]
        text_ignore=[]
        with open(os.path.join(labels_root,label_file),'r') as label:
            # label=file.read().encode('utf-8').decode('utf-8-sig')
            for box_with_text in label:
                text=box_with_text.split(',')[-1].strip('\n')
                box=box_with_text.split(',')[:-1]
                # boxs.append([[int(box[0].encode('utf-8').decode('utf-8-sig')),int(box[1])],[int(box[2]),int(box[3])],
                #              [int(box[4]),int(box[5])],[int(box[6]),int(box[7])]])
                boxs.append(
                    [[float(box[0].encode('utf-8').decode('utf-8-sig')), float(box[1])], [float(box[2]), float(box[3])],
                              [float(box[4]),float(box[5])],[float(box[6]),float(box[7])]])
                if text=='###':
                    text_ignore.append(True)
                else:
                    text_ignore.append(False)
            image_name=label_file[3:].split('.txt')[0]
            labels_dict[image_name] ={'boxs':boxs,'text_ignore':text_ignore}
            image_names.append(image_name)

    return image_names,labels_dict

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
        # show_kernels_mask(text_kernel_masks)
        # show_output_and_ground_true(output,text_kernel_masks,train_masks)
        # show_output_(output)
        # calculate the loss
        loss, loss_text, loss_kernel = pseloss(images, output, text_kernel_masks, train_masks,)

        # update the weight for the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logger info and tensorboard writer
        loggerinfo.add_step()
        if (index + 1) % config.TRAIN.SHOW_STEP == 0:
            loss_average, loss_text_average, loss_kernel_average = loggerinfo.average_loss()
            logger.info('epoch:[{}/{}],step:[{}/{}] all_steps:{}, loss:{:.4f}, loss_s:{:.4f}, loss_k:{:.4f}, {:.2f} samples/s, lr:{}'.
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
    return loss_average, loss_text_average, loss_kernel_average
def valid_one_epoch(config,psenet,pseloss,optimizer,schedular,writer,epoch):
    # pbar = tqdm(total=100)
    image_names, labels = load_imagename_labels(config.DATASET.LABEL_VAL_ROOT)
    len_images=len(image_names)
    long_edge=2240
    save_path='/home/simple/mydemo/ocr_project/pytorch_psenet_project/pytorch_psenet/results'
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    gt_path=config.DATASET.LABEL_VAL_ROOT
    for i in tqdm.tqdm(range(0,len_images)):
        image_name=image_names[i]+'.jpg'
        image=cv2.imread(os.path.join(config.DATASET.IMAGE_VAL_ROOT,image_name))
        image=cv2.cvtColor(image,code=cv2.COLOR_BGR2RGB)
        image_show=image.copy()
        # cv2.imshow('ttt',image_show)
        h,w,c=image.shape
        ratio=long_edge/max(h,w)
        image=cv2.resize(image,dsize=None,fx=ratio,fy=ratio)
        # image=torch.Tensor(image,dtype=torch.float32)
        image=torchvision.transforms.ToTensor()(image).unsqueeze(0).to(torch.device('cuda:' + config.CUDA.GPU))
        save_name = os.path.join(save_path, 'res_' + image_names[i] + '.txt')
        # put the batch dataset to GPU
        # images = images.to(torch.device('cuda:' + config.CUDA.GPU))
        # text_kernel_masks = text_kernel_masks.to(torch.device('cuda:' + config.CUDA.GPU))
        # train_masks = train_masks.to(torch.device('cuda:' + config.CUDA.GPU))

        # the model forward
        with torch.no_grad():
            output = psenet(image, True)
            preds, boxes_list = pse_decode(output[0], config.VALID.SCALE)
            scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')

    result_dict = cal_recall_precison_f1(gt_path, save_path)
    print(result_dict)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']
    # 关闭占用的资源
    # pbar.close()
    # loss_average, loss_text_average, loss_kernel_average = loggerinfo.average_loss()
    # logger.info(
    #     'epoch:[{}/{}],all_step:{}, loss:{:.4f}, loss_s:{:.4f}, loss_k:{:.4f}'.
    #         format(epoch, config.TRAIN.EPOCH, loggerinfo.step_epoch,
    #                loss_average[0], loss_text_average[0],
    #                loss_kernel_average[0]))
    # writer.add_scalar('Val/Loss', loss_average[0], epoch)
    # writer.add_scalar('Val/Loss_s', loss_text_average[0], epoch)
    # writer.add_scalar('Val/Loss_kernel', loss_kernel_average[0], epoch)

    #loggerinfo.clear()
# def show_output(output):
#     batch_size=output.shape[0]
#     N=output.shape[1]
#
#     for i in range(batch_size):
#         for n in range(N):
#             kernel=output[i,n,:,:].cpu().detach().numpy()>0.7
#             kernel=np.uint8(kernel)*255
#             cv2.imshow(str(n),kernel)
#     cv2.waitKey(20000)

def show_kernels_mask(kernels):
    batch_size = kernels.shape[0]
    N = kernels.shape[3]

    for i in range(batch_size):
        for n in range(N):
            kernel = kernels[i, :, :, n].cpu().detach().numpy()
            kernel = np.uint8(kernel) * 255
            cv2.imshow('groundtrue'+str(n), kernel)
        # cv2.waitKey(2000)
def show_output_and_ground_true(output,kernels,train_masks):
    batch_size = output.shape[0]
    N = output.shape[1]

    for i in range(batch_size):
        for n in range(N):
            if n==5 :
                kernel = output[i, n, :, :].cpu().detach().numpy() > 0.7
                kernel = np.uint8(kernel) * 255
                cv2.imshow(str(n), kernel)

                kernel = kernels[i, :, :, n].cpu().detach().numpy()
                kernel = np.uint8(kernel) * 255
                cv2.imshow('groundtrue' + str(n), kernel)

                kernel = train_masks[i, :, :].cpu().detach().numpy()
                kernel = np.uint8(kernel) * 255
                cv2.imshow('mask', kernel)
        cv2.waitKey(5000)
def show_output(output):
    batch_size=output.shape[0]
    N=output.shape[1]

    for i in range(batch_size):
        text_mask=output[i,5,:,:].cpu().detach().numpy()<0.5
        # cv2.imshow('text_mask', np.uint8(output[i,5,:,:].cpu().detach().numpy()<=0.5)*255)
        for n in range(0,N):


            kernel=output[i,n,:,:].cpu().detach().numpy()
            kernel[text_mask]=0
            kernel=kernel>0.5


            kernel=np.uint8(kernel*255)
            kernel=cv2.resize(kernel,dsize=None,fx=0.25,fy=0.25)
            cv2.imshow(str(n),kernel)
        cv2.waitKey(20000)