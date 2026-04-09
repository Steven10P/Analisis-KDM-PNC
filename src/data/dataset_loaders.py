# Archivo: src/data/dataset_loaders.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

class KDMDataPipelineMNISTKFold:
    def __init__(self, data_dir='./data', k_folds=5, random_state=42):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        self.data_dir = data_dir
        self.k_folds = k_folds
        self.kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=random_state)

    def load_full_numpy_datasets(self):
        train_set = datasets.MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        test_set = datasets.MNIST(self.data_dir, train=False, download=True, transform=self.transform)

        train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
        test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

        x_train, y_train = next(iter(train_loader))
        x_test, y_test = next(iter(test_loader))

        return x_train.numpy(), y_train.numpy(), x_test.numpy(), y_test.numpy()
        
    def get_splits(self, x_train_full):
        """Retorna el generador de índices para K-Fold"""
        return self.kf.split(x_train_full)
