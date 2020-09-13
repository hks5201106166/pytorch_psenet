#-*-coding:utf-8-*-
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from datasets.image_augment import DataAugment
from PIL import Image
import os
import cv2
import numpy as np
import pyclipper
import torch
class Dataset_PSE(Dataset):
    def __init__(self,config,train):
        self.augment=DataAugment()
        self.config=config
        self.train=train
        if train==True:
            self.image_names,self.labels=self.load_imagename_labels(config.DATASET.LABEL_TRAIN_ROOT)
        else:
            self.image_names,self.labels = self.load_imagename_labels(config.DATASET.LABEL_VAL_ROOT)

    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, index):
        image_name=self.image_names[index]+'.jpg'
        image=cv2.imread(os.path.join(self.config.DATASET.IMAGE_TRAIN_ROOT,image_name))
        image=cv2.cvtColor(image,code=cv2.COLOR_BGR2RGB)
        label=self.labels[self.image_names[index]]
        if self.train==True:
            image,boxs=self.augment_image(image,boxs=np.array(label['boxs']),config=self.config)
        else:
            boxs=np.array(label['boxs'],dtype=np.int)

        #gen the masks which is text ignonred
        boxs_text_no = []
        for box,text_ignore in zip(boxs,label['text_ignore']):
            if text_ignore==True:
                boxs_text_no.append(np.array(box).astype(np.int))
        # cv2.polylines(image,pts=boxs_text_yes,isClosed=True,color=(255,0,0),thickness=3)
        # cv2.polylines(image, pts=boxs_text_no, isClosed=True, color=(100, 100, 0),thickness=2)


        train_mask=np.ones(shape=(image.shape[0],image.shape[1]),dtype=np.uint8)
        cv2.fillPoly(train_mask,pts=boxs_text_no,color=(0))
        #gen the masks which is text and kernel
        text_kernel_masks=self.gen_label_text_kernel(image_shape=image.shape,boxs=boxs,n=self.config.MODEL.PSE.n,m=self.config.MODEL.PSE.m)
        if self.train==True:
            imgs=self.augment.random_crop_author([image,text_kernel_masks.transpose([1,2,0]),train_mask],img_size=(self.config.DATASET.IMAGE_SIZE.H,self.config.DATASET.IMAGE_SIZE.W))
            image=transforms.ToTensor()(Image.fromarray(imgs[0]))
            text_kernel_masks=imgs[1]
            train_mask=np.float32(imgs[2])
        else:
            image=transforms.ToTensor()(Image.fromarray(image))
            text_kernel_masks=np.float32(text_kernel_masks.transpose((1,2,0)))



        # cv2.imshow('image',image_crop)
        # cv2.imshow('kernel',text_kernel_masks[:,:,0])
        # cv2.imshow('train_mask',train_mask)
        # cv2.waitKey(10000)
        # train_mask=cv2.resize(train_mask,(0,0),fx=0.5,fy=0.5)
        # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        # cv2.imshow('ttt',train_mask)
        # cv2.imshow('image',image)
        # cv2.waitKey(5000)
        return image,text_kernel_masks,train_mask

    def augment_image(self,image,boxs,config):
        h_train,w_train=config.DATASET.IMAGE_SIZE.H,config.DATASET.IMAGE_SIZE.W
        image,boxs=self.augment.random_scale(image,scales=np.array([0.5,1.0,2.0,3.0]),text_polys=boxs)
        h,w,c=image.shape
        radio=max(h_train/h,w_train/w)
        if radio>1:
            boxs=boxs*radio
            image=cv2.resize(image,(0,0),fx=radio,fy=radio)
        flag=np.random.randn()
        if flag<0.5:
            image,boxs=self.augment.horizontal_flip(image,boxs)
        if flag<0.5:
            image,boxs=self.augment.random_rotate_img_bbox(image,boxs,degrees=10)

        return image,boxs.astype(np.int)
    def load_imagename_labels(self,labels_root):
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
    def gen_label_text_kernel(self,image_shape,boxs,n,m):

        label_text_kernel_masks=np.zeros(shape=(n,image_shape[0],image_shape[1]))
        for box in boxs:
            for i in range(1,n+1):
                r=1-(1-m)*(n-i)/(n-1)
                offset=cv2.contourArea(box)*(1-r*r)/cv2.arcLength(box,closed=True)
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                box_shrink = pco.Execute(-offset)
                cv2.fillPoly(label_text_kernel_masks[i-1,:,:],np.array(box_shrink),color=(1))
        # i=0
        # for label_text_kernel_mask in label_text_kernel_masks:
        #     cv2.imshow(str(i),label_text_kernel_mask)
        #     i+=1
        # cv2.waitKey(100000)

        return label_text_kernel_masks

    def check_and_validate_polys(self,polys, xxx_todo_changeme):
        '''
        check so that the text poly is in the same direction,
        and also filter some invalid polygons
        :param polys:
        :param tags:
        :return:
        '''
        (h, w) = xxx_todo_changeme
        if polys.shape[0] == 0:
            return polys
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)  # x coord not max w-1, and not min 0
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)  # y coord not max h-1, and not min 0

        validated_polys = []
        for poly in polys:
            p_area = cv2.contourArea(poly)
            if abs(p_area) < 1:
                continue
            validated_polys.append(poly)
        return np.array(validated_polys)


