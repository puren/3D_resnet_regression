from __future__ import print_function, division

import os
import sys
import json
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import time
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

from frankaDataset import FrankaDataset
from my_transform import *
from model import *
from train import train_epoch, val_epoch, test
from utils import Logger
from opts import parse_opts
from model import generate_model
from model_franka import *

#from mean import get_mean, get_std

if __name__ == '__main__':
  
  opt = parse_opts()
  
  opt.result_path = os.path.join(opt.root_path, opt.result_path)
  opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
  
  """Read info.xml file"""
  xml_dir = os.path.join(opt.root_path, 'info.xml')
  tree = ET.parse(xml_dir)
  root = tree.getroot()
  mean_rgb = np.array([0.0, 0.0, 0.0])
  std_rgb=np.array([0.0, 0.0, 0.0])
  mean_depth = 0.0
  std_depth = 0.0
  num_train = 0
  num_val = 0
  num_test = 0
  for type_img in root.findall('depth'):
    mean_d = float(type_img.find('mean_d').text)
    std_d = float(type_img.find('std_d').text)
  for type_img in root.findall('rgb'):
    mean_rgb[0] = float(type_img.find('mean_r').text)
    mean_rgb[1] = float(type_img.find('mean_g').text)
    mean_rgb[2] = float(type_img.find('mean_b').text)
    std_rgb[0] = float(type_img.find('std_r').text)
    std_rgb[1] = float(type_img.find('std_g').text)
    std_rgb[2] = float(type_img.find('std_b').text)
  for value in root.findall('width'):
    width = int(value.text)
  for value in root.findall('height'):
    height = int(value.text)
  for value in root.findall('num_js'):
    js_num = int(value.text)
  for value in root.findall('num_train'):
    num_train = int(value.text)
  for value in root.findall('num_val'):
    num_val = int(value.text)
  for value in root.findall('num_test'):
    num_test = int(value.text)

  if(not opt.pretrain_path):
     opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
     opt.n_classes = js_num * opt.sample_duration
  else:
     opt.n_finetune_classes = js_num * opt.sample_duration
  
  data_length = {'train': num_train, 'val': num_val, 'test': num_test}
  
  opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
  
  print(opt)
  with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
       json.dump(vars(opt), opt_file)
       
  mean_dataset = []
  std_dataset = []
  inputs = []
  if(opt.is_rgb and not opt.is_depth):
    mean_dataset = mean_rgb
    std_dataset = std_rgb
  elif(not opt.is_rgb and opt.is_depth):
    mean_dataset = mean_d
    std_dataset = std_d
  if(opt.is_rgb and opt.is_depth):
    mean_dataset = np.array([mean_rgb[0], mean_rgb[1], mean_rgb[2], mean_depth])
    std_dataset = np.array([std_rgb[0], std_rgb[1], std_rgb[2], std_depth])
  
        
  torch.manual_seed(opt.manual_seed)
  
  
  model, parameters = generate_model(opt)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model.to(device)
    
  criterion = nn.MSELoss()
  criterion = criterion.cuda()
  
  transform_i=[]
  transform_t=[]
  if(opt.is_scale):
    transform_i = transforms.Compose([Normalize(mean_dataset, std_dataset), Rescale((opt.sample_size, opt.sample_size)),ToTensor()])
  else:
    transform_i = transforms.Compose([Normalize(mean_dataset, std_dataset), ToTensor()])
  
  transform_t = ToTensor()
  
  trainset = FrankaDataset(opt, 'train', data_length, transform_input=transform_i, transform_target=transform_t)
  trainloader = DataLoader(trainset, 
                              batch_size=opt.batch_size, 
                              shuffle=True,
                              num_workers=opt.n_threads, 
                              pin_memory=True)
  if opt.is_train:
     
     trainset = FrankaDataset(opt, 'train', data_length, transform_input=transform_i, transform_target=transform_t)
     trainloader = DataLoader(trainset, 
                              batch_size=opt.batch_size, 
                              shuffle=False,
                              num_workers=opt.n_threads, 
                              pin_memory=True)
     train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'err', 'lr'])
     train_batch_logger = Logger(
        os.path.join(opt.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'err', 'lr'])
     
     optimizer = optim.SGD(parameters, 
                           lr=opt.learning_rate, 
                           momentum=opt.momentum, 
                           dampening=opt.dampening,
                           weight_decay=opt.weight_decay,
                           nesterov=opt.nesterov)
     scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)

  if opt.is_val:
      
      valset = FrankaDataset(opt, 'val', data_length, transform_input=transform_i, transform_target=transform_t)
      valloader = DataLoader(valset, 
                              batch_size=opt.batch_size,
                              shuffle=False, 
                              num_workers=opt.n_threads, 
                              pin_memory=True)
      val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'err'])
     
  if opt.resume_path:
      print('loading checkpoint {}'.format(opt.resume_path))
      checkpoint = torch.load(opt.resume_path)
      assert opt.arch == checkpoint['arch']


      opt.begin_epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['state_dict'])
      if opt.is_train:
          optimizer.load_state_dict(checkpoint['optimizer'])   


  for epoch in range(opt.begin_epoch, opt.n_epochs + 1):
      
      if opt.is_train:
          train_epoch(epoch, train_logger, train_batch_logger, trainloader, optimizer, criterion, model, opt, device)
          
      if opt.is_val:
          validation_loss = val_epoch(epoch, val_logger, valloader, optimizer, criterion, model, device)
                    
      if opt.is_train and opt.is_val:
             scheduler.step(validation_loss)
  
      if opt.is_test:
     
        testset = FrankaDataset(opt, 'test', data_length, transform_input=transform_i, transform_target=transform_t)
        testloader = DataLoader(testset, 
                              batch_size=opt.batch_size,
                              shuffle=False, 
                              num_workers=opt.n_threads, 
                              pin_memory=True)
        test(trainloader, model, device)
        
         
         
                      
         
