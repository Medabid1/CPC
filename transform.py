import torch 

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms




class overlap:
    def __init__(self, size):
        self.h , self.w = size, size
    
    def __call__(self, img): 
        assert img.size(2) % self.h == 0, 'Image height must be divided by size[0]'
        assert img.size(1) % self.w == 0, 'Image height must be divided by size[1]'
        
        imgs = []
        for i in range(img.size(2)//self.h + 1):
            for j in range(img.size(1)//self.w + 1):
                imgs.append(img[:,i * self.h//2: i * self.h//2 + self.h , j * self.w//2 :self.w + j * self.w//2])
        imgs = torch.stack(imgs, dim=0)

        return imgs
