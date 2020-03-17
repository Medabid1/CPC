import torch 

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from model import CPCModel



mnist_dataset = MNIST('data')