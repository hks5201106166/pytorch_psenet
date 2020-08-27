#-*-coding:utf-8-*-
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from datasets.image_augment import DataAugment
import os
import cv2
import numpy as np
import pyclipper
class Dataset_PSE(Dataset):
    def __init__(self,config,train_or_val='train'):
        self.augment=DataAugment()
        self.config=config
        if train_or_val=='train':
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
        image,boxs=self.augment_image(image,boxs=np.array(label['boxs']),config=self.config)

        #gen the masks which is text ignonred
        boxs_text_no = []
        for box,text_ignore in zip(label['boxs'],label['text_ignore']):
            if text_ignore==True:
                boxs_text_no.append(np.array(box))
        # cv2.polylines(image,pts=boxs_text_yes,isClosed=True,color=(255,0,0),thickness=3)
        cv2.polylines(image, pts=boxs_text_no, isClosed=True, color=(0, 0, 0),thickness=2)
        train_mask=255*np.ones(shape=(image.shape[0],image.shape[1]),dtype=np.uint8)
        cv2.fillPoly(train_mask,pts=boxs_text_no,color=(0))
        # cv2.fillPoly(image,,pts=boxs_text_no,color=(0))

        #gen the masks which is text and kernel
        text_kernel_mask=self.gen_label_text_kernel(label['boxs'],label['text_ignore'],5,0.5)
        # cv2.imshow('ttt',train_mask)
        # cv2.imshow('image',image)
        # cv2.waitKey(10000)
        return image,text_kernel_mask,train_mask

    def augment_image(self,image,boxs,config):
        h_train,w_train=config.DATASET.IMAGE_SIZE.H,config.DATASET.IMAGE_SIZE.W
        image,boxs=self.augment.random_scale(image,scales=np.array([0.5,1.0,2.0,3.0]),text_polys=boxs)
        h,w,c=image.shape
        l_min=min(h,w)
        if l_min<

        return image,boxs
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
    def gen_label_text_kernel(self,boxs,text_ignored,n,m):
        label_text_kernel_mask=0
        return label_text_kernel_mask


