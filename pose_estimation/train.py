# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import copy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Subset 
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from core.function import train_dann
from core.function import train_src
from core.function import train_tgt
from core.function import eval_src
from core.function import eval_tgt
#from core.function import train_tgt
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
import torch.nn as nn

import dataset
import models
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    ################# AGGIUNTO #######################
    parser.add_argument('--adapt',
                        help='1 for using DANN, 2 for using ADDA',
                        type=int)
	##################################

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers

def train_normale(DANN,args,config):
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    ################# AGGIUNTO #######################
    criterion_dann = nn.CrossEntropyLoss()
    ########################################
    
    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    ################# AGGIUNTO #######################
    if DANN==True:
      painting_dataset = eval('dataset.painting')(
          config,
          config.DATASET.ROOT,
          config.DATASET.PAINTING_SET,
          False,
          transforms.Compose([
              #transforms.Resize(256),
              #transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
          ])
      )
      
      train_no_aug_dataset = eval('dataset.painting')(
          config,
          config.DATASET.ROOT,
          config.DATASET.TRAIN_NO_AUG_SET,
          False,
          transforms.Compose([
              #transforms.Resize(256),
              #transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
          ])
      )
      
	##################################
      

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    ################# AGGIUNTO #######################
    if DANN==True:
      painting_loader = torch.utils.data.DataLoader(
          painting_dataset,
          #valid_dataset,
          batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
          shuffle=config.TRAIN.SHUFFLE,
          num_workers=config.WORKERS,
          pin_memory=True
      )
      #painting_loader=copy.deepcopy(valid_loader)
      
      train_no_aug_loader = torch.utils.data.DataLoader(
          train_no_aug_dataset,
          #valid_dataset,
          batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
          shuffle=config.TRAIN.SHUFFLE,
          num_workers=config.WORKERS,
          pin_memory=True
      )
	##################################
    

    best_perf = 0.0
    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        
        # train for one epoch
        ################# AGGIUNTO #######################
        if DANN==True:
            alpha=0.03
            print(f"Alpha = {alpha}")
            model=train_dann(config, train_loader, train_no_aug_loader, painting_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict, alpha, criterion_dann)
        else:
        	train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)
        ##################################


        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, tb_log_dir,
                                  writer_dict)
        lr_scheduler.step()

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
        

def train_adda(args,config):
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    source_encoder = eval('models.'+config.MODEL.NAME+'.get_pose_net_encoder')(
        config, is_train=True
    )
    target_encoder = eval('models.'+config.MODEL.NAME+'.get_pose_net_encoder')(
        config, is_train=True
    )
    pose_estimator = eval('models.'+config.MODEL.NAME+'.get_pose_net_deconv')(
        config, is_train=True
    )
    critic = models.ADDA_discriminator.get_discriminator()
    
    criterion_adda = nn.CrossEntropyLoss()
    
    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL.NAME + '.py'),
        final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((config.TRAIN.BATCH_SIZE,
                             3,
                             config.MODEL.IMAGE_SIZE[1],
                             config.MODEL.IMAGE_SIZE[0]))
    #writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    source_encoder = torch.nn.DataParallel(source_encoder, device_ids=gpus).cuda()
    target_encoder = torch.nn.DataParallel(target_encoder, device_ids=gpus).cuda()
    pose_estimator = torch.nn.DataParallel(pose_estimator, device_ids=gpus).cuda()
    critic = torch.nn.DataParallel(critic, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    
    #optimizer = get_optimizer(config, pose_estimator)

    optimizer = optim.Adam(
        list(source_encoder.parameters()) + list(pose_estimator.parameters()),
        lr=config.TRAIN.LR,
    )
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TRAIN_SET,
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_set = Subset(train_dataset,train_indices)
    train_valid_set = Subset(train_dataset,val_indices)

    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    painting_dataset = eval('dataset.painting')(
        config,
        config.DATASET.ROOT,
        config.DATASET.PAINTING_SET,
        False,
        transforms.Compose([
              #transforms.Resize(256),
              #transforms.CenterCrop(224),
              transforms.ToTensor(),
              normalize,
        ])
      )
      

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    train_valid_loader = torch.utils.data.DataLoader(
        train_valid_set,
        batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    
    painting_loader = torch.utils.data.DataLoader(
          painting_dataset,
          batch_size=config.TRAIN.BATCH_SIZE*len(gpus),
          shuffle=config.TRAIN.SHUFFLE,
          num_workers=config.WORKERS,
          pin_memory=True
    )
    #painting_loader=copy.deepcopy(valid_loader)
    
    ###################### TRAIN SOURCE ENCODER AND POSE ESTIMATOR ######################
    '''
    print("(1.a)-----------------Training pose estimator for source")
    best_perf = 0.0
    best_model = False
    best_source_encoder=None
    best_pose_estimator=None
    
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
      
        source_encoder, pose_estimator = train_src(config, train_loader, source_encoder, pose_estimator, criterion, optimizer, epoch, final_output_dir, tb_log_dir)


        # evaluate on validation set
        #perf_indicator = eval_src(config, train_loader, train_set,
        eval_src(config, train_loader, train_set,
                                  source_encoder, pose_estimator,
                                  criterion, final_output_dir, tb_log_dir)
        lr_scheduler.step()
    
        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
            best_source_encoder=copy.deepcopy(source_encoder)
            best_pose_estimator=copy.deepcopy(pose_estimator)
        else:
            best_model = False
		
        
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()
    
    print("(1.b)----------------Evaluation source encoder")
    
    eval_src(config, train_valid_loader, train_valid_set, 
                                  source_encoder, pose_estimator,
                                  criterion, final_output_dir, tb_log_dir)
    
    '''
    ###################### TRAIN TARGET ENCODER ######################
    print("(2)----------------Training target encoder")
  
    target_encoder.load_state_dict(source_encoder.state_dict())
    target_encoder = train_tgt(source_encoder, target_encoder, critic,
                                train_loader, painting_loader,config)
    
    
    ###################### TEST ######################
    print("(3)----------------Test")    
    print("(3.a)---------------->>> source only <<<")
    eval_tgt(config, valid_loader, valid_dataset, 
             source_encoder, pose_estimator,
             criterion, final_output_dir, tb_log_dir)
    
    print("(3.b)---------------->>> domain adaption <<<")
    eval_tgt(config, valid_loader, valid_dataset, 
             target_encoder, pose_estimator,
             criterion, final_output_dir + '2', tb_log_dir)
    
def main():
    DANN = False
    ADDA = False
    args = parse_args()
    reset_config(config, args)
    ################# AGGIUNTO #######################
    if args.adapt==1:
        DANN = True
    elif args.adapt==2:
        ADDA = True
	##################################

    if ADDA==False:
      train_normale(DANN,args,config)
    else:
      train_adda(args,config)
    


if __name__ == '__main__':
    main()
