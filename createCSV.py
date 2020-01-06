import sys
import os
import xml.etree.ElementTree as ET
from random import shuffle
import glob
from scipy import misc, io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import errno
import time
import csv
import argparse


class csv_data:
  def __init__(self, root_dir):
    
    self.mu_depth = 0
    self.std_depth = 0
    self.mu_rgb = np.zeros(3)
    self.std_rgb = np.zeros(3)
    self.height = 0
    self.width = 0
    self.js_num = 0
    
    self.mu_depth_new = 0
    self.mu_depth_curr = 0 
    self.var_depth_new = 0
    self.var_depth_curr = 0
    self.mu_rgb_new = 0
    self.mu_rgb_curr = np.zeros(3)
    self.var_rgb_new = np.zeros(3)
    self.var_rgb_curr = np.zeros(3)
    self.curr_size = 0
    self.total_size = 0
    self.img_size = 0
    
    self.numTrain=0
    self.numVal=0
    self.numTest=0
    self.numTest_seq=0
    self.root_dir=root_dir
    
    self.rgb_addrs = []
    self.depth_addrs = []
    self.joint_addrs = []
    self.indices = []
  
  
  @staticmethod
  def load_depth_image( addr):
    img = misc.imread(addr)

    if img is None:
        print("Could not open or find the image:")
        print(addr)
        sys.exit()
    img = img.astype(np.float32)
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            
    return img
  
  @staticmethod
  def load_rgb_image( addr):
    img = misc.imread(addr)
      
    if img is None:
        print("Could not open or find the image:")
        print(addr)
        sys.exit()
    img = img.astype(np.uint8)
    
    return img
  

  # here, the ground truth data for joints is read.
  @staticmethod
  def load_joint(addr):
    f = open(addr)
    if f is None:
        print("Could not open or find the the file:")
        print(addr)
        sys.exit()
    js = [[num for num in line.split(' ')] for line in f ]
    joints = js[0][0:-1]           
    joints = np.array(joints).astype(np.float32)
    
    return joints
  
  @staticmethod
  def update_mean(self, img):
      if(img.shape[2]==1):
        self.mu_depth_new=np.mean(img)
        return self.mu_depth_curr*(float(self.curr_size)/self.total_size) + self.mu_depth_new*(float(self.img_size)/self.total_size)
      elif(img.shape[2]>1):
        self.mu_rgb_new = np.zeros(3)
        self.mu_rgb_new[0]=np.mean(img[:,:,0])
        self.mu_rgb_new[1]=np.mean(img[:,:,1])
        self.mu_rgb_new[2]=np.mean(img[:,:,2])
        return self.mu_rgb_curr*(float(self.curr_size)/self.total_size) + self.mu_rgb_new*(float(self.img_size)/self.total_size)
      
  @staticmethod
  def update_var(self, img):
      if(img.shape[2]==1):
        self.var_depth_new = np.var(img)
        return self.var_depth_new*(float(self.img_size)/self.total_size) + self.var_depth_curr*(float(self.curr_size)/self.total_size) + (float(self.img_size*self.curr_size)/pow(self.total_size, 2))*pow((self.mu_depth_new-self.mu_depth_curr), 2)
      elif(img.shape[2]>1):
        self.var_rgb_new = np.zeros(3)
        self.var_rgb_new[0]=np.var(img[:,:,0])
        self.var_rgb_new[1]=np.var(img[:,:,1])
        self.var_rgb_new[2]=np.var(img[:,:,2])
        return self.var_rgb_new*(float(self.img_size)/self.total_size) + self.var_rgb_curr*(float(self.curr_size)/self.total_size) + (float(self.img_size*self.curr_size)/pow(self.total_size, 2))*pow((self.mu_rgb_new-self.mu_rgb_curr), 2)
   
   
  def split_dataset(self, start_ind, end_ind):
      indices = self.indices[start_ind:end_ind]
      rgb_addrs = self.rgb_addrs[start_ind:end_ind]
      depth_addrs = self.depth_addrs[start_ind:end_ind]
      joint_addrs = self.joint_addrs[start_ind:end_ind]
      
      return indices, rgb_addrs, depth_addrs, joint_addrs
      
  def writeToCSVFile(self, indices, rgb_addrs, depth_addrs, joint_addrs, N, mode):
   
    var_rgb_curr=np.zeros(3)
    var_depth_curr=0
    std_depth_curr=0
    std_rgb_curr=np.zeros(3)
    mu_depth_curr=0
    mu_rgb_curr=np.zeros(3)
    img_size=0
    
    
    count=0
    
    with open(os.path.join(self.root_dir, '{}.csv'.format(mode)), 'wb') as csvfile:
      csv_writer = csv.writer(csvfile)
      header = ['indx', 'rgb_path', 'depth_path', 'js_path']
      csv_writer.writerow(header)
      
      self.curr_size = 0
      self.total_size = 0
      for i in range(0, N):
        # print how many data points are saved every 1000 images
        if not count % 1000:
            print('Data: {}/{}/{}'.format(count, i, N))
            sys.stdout.flush()
            
        # Load the image
        depth_img = csv_data.load_depth_image('{}/{}'.format(self.root_dir, depth_addrs[i]))
        rgb_img = csv_data.load_rgb_image('{}/{}'.format(self.root_dir, rgb_addrs[i]))
        joint = csv_data.load_joint('{}/{}'.format(self.root_dir, joint_addrs[i]))
        
          
        if(i==0):
          self.height = depth_img.shape[0]
          self.width = depth_img.shape[1]
          self.js_num = len(joint)
          self.img_size=self.height *self.width
        
        addrs = [indices[i], rgb_addrs[i], depth_addrs[i], joint_addrs[i]]
        csv_writer.writerow(addrs) 
        
        #If train, calculate mean and std.
        if(mode=="train"):
          self.curr_size = count*self.img_size
          self.total_size=self.curr_size+self.img_size
          
          #mu_depth_new=np.mean(depth_img)
          #mu_depth_curr = mu_depth_curr*(float(curr_size)/total_size) + mu_depth_new*(float(img_size)/total_size)
          self.mu_depth_curr = csv_data.update_mean(self, depth_img)
          #self.var_depth_new=np.var(depth_img)
          #self.var_depth_curr = self.var_depth_new*(float(self.img_size)/self.total_size) + self.var_depth_curr*(float(self.curr_size)/self.total_size) + (float(self.img_size*self.curr_size)/pow(self.total_size, 2))*pow((self.mu_rgb_new-self.mu_rgb_curr), 2)
          self.var_depth_curr = csv_data.update_var(self, depth_img)
          
          """
          mu_rgb_new = np.zeros(3)
          mu_rgb_new[0]=np.mean(rgb_img[:,:,0])
          mu_rgb_new[1]=np.mean(rgb_img[:,:,1])
          mu_rgb_new[2]=np.mean(rgb_img[:,:,2])
          mu_rgb_curr = mu_rgb_curr*(float(curr_size)/total_size) + mu_rgb_new*(float(new_size)/total_size)
          
          var_rgb_new = np.zeros(3)
          var_rgb_new[0]=np.var(rgb_img[:,:,0])
          var_rgb_new[1]=np.var(rgb_img[:,:,1])
          var_rgb_new[2]=np.var(rgb_img[:,:,2])
          var_rgb_curr = var_rgb_new*(float(new_size)/total_size) + var_rgb_curr*(float(curr_size)/total_size) + (float(new_size*curr_size)/pow(total_size, 2))*pow((mu_rgb_new-mu_rgb_curr), 2)
          """
          self.mu_rgb_curr = csv_data.update_mean(self, rgb_img)
          self.var_rgb_curr =csv_data.update_var(self, rgb_img) 
          
        count=count+1
        
    if(mode=="train"):
      self.numTrain = count
      self.mu_depth = self.mu_depth_curr
      self.std_depth = np.sqrt(self.var_depth_curr)
      self.mu_rgb = self.mu_rgb_curr
      self.std_rgb = np.sqrt(self.var_rgb_curr)
    elif(mode=="val"):
      self.numVal=count
    elif(mode=="test"):
      self.numTest=count
      
      

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument(
        '--perc_train',
        default=60,
        type=int,
        help=
        'percentage of training data'
    )
  parser.add_argument(
        '--perc_val',
        default=20,
        type=int,
        help=
        'percentage of val data'
    )
  parser.add_argument(
        '--perc_test',
        default=20,
        type=int,
        help=
        'percentage of test data'
    )
  parser.add_argument(
        '--data_dir',
        default='/data/',
        type=str,
        help='Root directory path of data')
  args = parser.parse_args()
  
  
  # get data directory for training and testing
  path_data = args.data_dir
  num_train = float(args.perc_train)/100
  num_val = float(args.perc_val)/100
  num_test = float(args.perc_test)/100
  
  # read addresses and labels from the 'train' folder
  addrs = sorted(glob.glob(path_data + '/test_depth_*.png'))

  fd = csv_data(path_data)
    
  for addr_i in addrs:
    indx = (addr_i.split("_")[-1]).split(".")[0]
    fd.indices.append(indx)
    fd.rgb_addrs.append('test_rgb_{}.png'.format(indx))
    fd.depth_addrs.append('test_depth_{}.png'.format(indx))
    fd.joint_addrs.append('test_js_{}.txt'.format(indx))
  
  
  org_len = len(fd.rgb_addrs)
  # to shuffle rest of the data
  #c = list(zip(self.indices, self.rgb_addrs, self.depth_addrs, self.joint_addrs))
  #shuffle(c)
  #self.indices, self.rgb_addrs, self.depth_addrs, self.joint_addrs = zip(*c)
  
  
  start_ind = 0
  end_ind = int(num_train*org_len)
  """
  train_indices = indices[start_ind:end_ind]
  train_rgb_addrs = rgb_addrs[start_ind:end_ind]
  train_depth_addrs = depth_addrs[start_ind:end_ind]
  train_label_addrs = label_addrs[start_ind:end_ind]
  train_joint_addrs = joint_addrs[start_ind:end_ind]
  print(len(train_indices))
  """
  train_indices, train_rgb_addrs, train_depth_addrs, train_joint_addrs = fd.split_dataset(start_ind, end_ind)
  
  start_ind = end_ind+1
  end_ind = start_ind + int(num_val*org_len)
  val_indices, val_rgb_addrs, val_depth_addrs, val_joint_addrs = fd.split_dataset(start_ind, end_ind)
  
  
  start_ind = end_ind+1
  end_ind = org_len
  test_indices, test_rgb_addrs, test_depth_addrs, test_joint_addrs = fd.split_dataset(start_ind, end_ind)
  
  
  
  N = len(train_indices)
  M = len(val_indices)
  T = len(test_indices)
  
  # write in the csv file
  print("train")
  count_train=fd.writeToCSVFile(train_indices, train_rgb_addrs, train_depth_addrs, train_joint_addrs, N, "train")
  print("validation")
  fd.writeToCSVFile(val_indices, val_rgb_addrs, val_depth_addrs, val_joint_addrs, M, "val")
  print("testing")
  fd.writeToCSVFile(test_indices, test_rgb_addrs, test_depth_addrs, test_joint_addrs, T, "test")
      
  ######### write to xml ##################
  data = ET.Element('data')  
  num_train = ET.SubElement(data, 'num_train')
  num_train.text = str(fd.numTrain)
  num_val = ET.SubElement(data, 'num_val')
  num_val.text = str(fd.numVal)
  num_test = ET.SubElement(data, 'num_test')
  num_test.text = str(fd.numTest)
  img_width = ET.SubElement(data, 'width')
  img_width.text = str((fd.width))
  img_height = ET.SubElement(data, 'height')
  img_height.text = str((fd.height))
  num_js = ET.SubElement(data, 'num_js')
  num_js.text = str((fd.js_num))

  data_type = ET.SubElement(data, 'depth')
  mean = ET.SubElement(data_type, 'mean_d') 
  std = ET.SubElement(data_type, 'std_d')
  mean.text = str(fd.mu_depth)  
  std.text = str(fd.std_depth)
  
  data_type = ET.SubElement(data, 'rgb')
  mean_r = ET.SubElement(data_type, 'mean_r') 
  mean_r.text = str(fd.mu_rgb[0])  
  mean_g = ET.SubElement(data_type, 'mean_g') 
  mean_g.text = str(fd.mu_rgb[1])  
  mean_b = ET.SubElement(data_type, 'mean_b') 
  mean_b.text = str(fd.mu_rgb[2])  
  
  std_r = ET.SubElement(data_type, 'std_r') 
  std_r.text = str(fd.std_rgb[0])  
  std_g = ET.SubElement(data_type, 'std_g') 
  std_g.text = str(fd.std_rgb[1])  
  std_b = ET.SubElement(data_type, 'std_b') 
  std_b.text = str(fd.std_rgb[2])  
  
  mydata = ET.tostring(data)
  xml_dir = os.path.join(path_data, 'info_csv.xml')
  myfile = open(xml_dir, "wb")  
  myfile.write(mydata)
  myfile.close()
  


 

