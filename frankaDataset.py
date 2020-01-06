import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from skimage import io, transform
import os
from PIL import Image

class FrankaDataset(Dataset):

    def __init__(self, opt, mode, data_length, transform_input=None, transform_target=None):
        """
        Args:
           csv_file (string): Path to the csv file with annotations.
           root_dir (string): Directory with all the images.
           transform (callable, optional): Optional transform to be applied on a sample.
        """

        csv_data = pd.read_csv(os.path.join(opt.root_path, '{}.csv'.format(mode)))
        self.csv_indx = list(csv_data.iloc[:, 0])
        self.csv_rgb = list(csv_data.iloc[:, 1])
        self.csv_depth = list(csv_data.iloc[:, 2])
        self.csv_js = list(csv_data.iloc[:, 3])
        
        self.root_dir = opt.root_path
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.is_rgb = opt.is_rgb
        self.is_depth = opt.is_depth
        self.sample_duration = opt.sample_duration
        self.data_length = data_length[mode]

    def __len__(self):
        return len(self.csv_indx)
 
    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
         #  idx = idx.tolist()
        
        inputs = []
        js_l = []
        indx_l = []
        
        count=0
        it=self.csv_indx[idx]
        while count < self.sample_duration:  
         
          it_cur = it + count
          if(it_cur not in self.csv_indx):
             continue
             
          count = count+1
          
          ind = self.csv_indx.index(it_cur)
          #rgb_path = os.path.join(self.root_dir, str(self.ind.iloc[it, 1]))
          rgb_path = os.path.join(self.root_dir, str(self.csv_rgb[ind]))
          depth_path = os.path.join(self.root_dir, str(self.csv_depth[ind]))
          js_path = os.path.join(self.root_dir, str(self.csv_js[ind]))
          
          if self.is_rgb and not self.is_depth:
            rgb_img = Image.open(rgb_path)
            if rgb_img is None:
              print("Could not open or find the image:{}".format(rgb_img))
              sys.exit()
            rgb_img = rgb_img.convert(mode='RGB')
            input_i = np.array(rgb_img)
          elif self.is_depth and not self.is_rgb:
            depth_img = Image.open(depth_path)
            if depth_img is None:
              print("Could not open or find the image:{}".format(depth_img))
              sys.exit()
            depth_img = depth_img.convert(mode='F')
            input_i = np.array(depth_img)
            
          elif self.is_depth and self.is_rgb:
            depth_img = Image.open(depth_path)
            if depth_img is None:
              print("Could not open or find the image:{}".format(depth_img))
              sys.exit()
            depth_img = depth_img.convert(mode='F')
            input_i = np.array(depth_img)
            
            rgb_img = Image.open(rgb_path)
            if rgb_img is None:
              print("Could not open or find the image:{}".format(rgb_img))
              sys.exit()
            rgb_img = rgb_img.convert(mode='RGB')
            input_i = np.array(rgb_img)
            
            input_i = np.stack([rgb_img, dept_img], axis=0)
            
          f_js = open(js_path)
          if f_js is None:
             print("Could not open or find the the file:i{}".format(js_path))
             sys.exit()
          js = [[num for num in line.split(' ')] for line in f_js ]
          joints = np.array(js[0][0:-1]).astype(np.float32) 
           
          inputs.append(input_i)
          js_l.append(joints) 
          indx_l.append(ind) 
        
        js_l = np.array(js_l).flatten()
        
        if self.transform_input:
           inputs =[self.transform_input(img) for img in inputs]
        if self.transform_target:
           js_l = self.transform_target(js_l) 
        
        inputs = torch.stack(inputs, 0).permute(3, 0, 1, 2)           
        
        sample = {'inputs': inputs, 
                  'js': js_l,
                  'indx': indx_l
                 }
        
        return sample

 
