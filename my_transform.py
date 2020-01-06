from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image 
 
class Rescale(object):
    
    #It can rescale the image in a sample to any given size.
    #Args:
    #    output_size (tuple or int): Desired output size. If tuple, output is matched to output_size. If int, smaller of image edges is matched to output_size keeping aspect ratio the same.
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    
    def __call__(self, sample):
    
        image = sample
        
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        
        new_h, new_w = int(new_h), int(new_w)
        
        image = transform.resize(image, (new_h, new_w), mode='constant', anti_aliasing=True, anti_aliasing_sigma=None)
        
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        sample = torch.from_numpy(sample).float() 
        
        return sample
   
        
class Normalize(object):

    def __init__(self, mean, std):
        #assert isinstance(mean_d, (float, tuple))
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
    
        image = sample
        
        image = ( (image - self.mean)/self.std).astype(np.float32)
        
        return image
                
  

                  
