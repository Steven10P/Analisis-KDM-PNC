import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from keras.datasets import mnist
import numpy as np
from .base_pipeline import BaseDataPipeline

class MNISTPipeline(BaseDataPipeline):
    def __init__(self, data_dir='./data', batch_size=32):
        super().__init__(data_dir, batch_size)
        self.load_data()
    def load_data(self):
        (self.X_train_kdm, self.y_train_kdm), (self.X_test_kdm, self.y_test_kdm) = mnist.load_data()
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_set_pnc = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True, transform=transform)
        self.test_set_pnc = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True, transform=transform)
    def get_kdm_data(self):
        X_train = self.X_train_kdm.reshape(-1, 784).astype('float32') / 255.0
        X_test = self.X_test_kdm.reshape(-1, 784).astype('float32') / 255.0
        return X_train, self.y_train_kdm, X_test, self.y_test_kdm
    def get_pnc_loaders(self):
        test_loader = DataLoader(self.test_set_pnc, batch_size=self.batch_size, shuffle=False)
        return self.train_set_pnc, test_loader

def get_pipeline(dataset_name, batch_size=32):
    if dataset_name.lower() == 'mnist': return MNISTPipeline(batch_size=batch_size)
    else: raise ValueError(f'Dataset {dataset_name} no implementado.')
