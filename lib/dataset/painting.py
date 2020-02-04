# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints
from PIL import Image

logger = logging.getLogger(__name__)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #img = Image.open(f)
        img = np.array(Image.open(f))
        im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #return img.convert('RGB')
        return Image.fromarray(im_bgr)
    
class PaintingDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        #self.num_joints = 0
        #self.pixel_std = 200
        #self.flip_pairs = []
        #.parent_ids = []

        #self.is_train = is_train
        self.root = root
        self.image_set = image_set

        #self.output_path = cfg.OUTPUT_DIR
        #self.data_format = cfg.DATASET.DATA_FORMAT

        #self.scale_factor = cfg.DATASET.SCALE_FACTOR
        #self.rotation_factor = cfg.DATASET.ROT_FACTOR
        #self.flip = cfg.DATASET.FLIP

        #self.image_size = cfg.MODEL.IMAGE_SIZE
        #self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        #self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        #self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        
        images_dir = os.path.join(self.root + 'images', self.image_set)
        print("Images dir " + images_dir)
        self.db = []
               
        for image_name in os.listdir(images_dir):
            image_full_path = os.path.join(images_dir, image_name)
            self.db.append(image_full_path)
        
		
    #def _get_db(self):
    #    raise NotImplementedError

    #def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
    #    raise NotImplementedError

    def __len__(self,):
        return len(self.db)
      
    
    def __getitem__(self, idx):
        #db_rec = copy.deepcopy(self.db[idx])
        #print("GET ITEM")
        #image_file = db_rec['image']
        image_file = self.db[idx]
        #filename = db_rec['filename'] if 'filename' in db_rec else ''
        #imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        #if self.data_format == 'zip':
        #    from utils import zipreader
        #    data_numpy = zipreader.imread(
        #        image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        #else:
        
        data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        #data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        #image=Image.fromarray(data_numpy)
        
        #image = pil_loader(image_file)

        
        width = 192
        height = 256
        dim = (width, height) 
        resized = cv2.resize(data_numpy, dim)
        
        
        '''r = 256.0 / data_numpy.shape[1]
        dim = (256, int(data_numpy.shape[0] * r))
 
        # perform the actual resizing of the image and show it
        resized = cv2.resize(data_numpy, dim, interpolation = cv2.INTER_AREA)'''
        
        '''c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
		
        if self.transform:
            input = self.transform(input)'''
		
        if self.transform is not None:
            input = self.transform(resized)
        #else:
        #    input = data_numpy
            
        meta = {
            'image': image_file}

        return input, meta

    '''def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected'''

    
