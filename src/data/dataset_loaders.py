# Archivo: src/data/dataset_loaders.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

import os
import urllib.request
import scipy.io as sio
from tensorflow.keras.utils import to_categorical

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


class PNCDataPipelineKFold:
    """
    Pipeline de datos nativo de PyTorch para Probabilistic Neural Circuits.
    Aplica transformaciones de tensores y gestiona los DataLoaders para K-Fold.
    """
    def __init__(self, batch_size=128, data_dir='./data'):
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        print("[INFO] Descargando/Cargando datasets MNIST (Tensores PyTorch)...")
        self.train_set_full = datasets.MNIST(data_dir, train=True, download=True, transform=self.transform)
        self.test_set_full = datasets.MNIST(data_dir, train=False, download=True, transform=self.transform)

    def get_fold_loaders(self, train_idx, val_idx):
        """Genera iteradores para el fold actual."""
        train_sub = Subset(self.train_set_full, train_idx)
        val_sub = Subset(self.train_set_full, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def get_test_loader(self):
        """Genera el iterador para evaluación final."""
        return DataLoader(self.test_set_full, batch_size=self.batch_size, shuffle=False)

# ==========================================
# NUEVA CLASE PARA FASHION-MNIST
# ==========================================

class FashionPNCDataPipelineKFold:
    def __init__(self, batch_size=128, data_dir='./data_fashion'):
        self.batch_size = batch_size
        
        # Mantenemos SOLO ToTensor() para no romper el escalado (data * 255.0).long()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

        print("Cargando datasets Fashion-MNIST...")
        self.train_set_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=self.transform)
        self.test_set_full = datasets.FashionMNIST(data_dir, train=False, download=True, transform=self.transform)

    def get_fold_loaders(self, train_idx, val_idx):
        train_sub = Subset(self.train_set_full, train_idx)
        val_sub = Subset(self.train_set_full, val_idx)
        train_loader = DataLoader(train_sub, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader


class FashionKDMDataPipelineKFold:
    def __init__(self, data_dir='./data_fashion'):
        """
        Pipeline de datos MLOps para KDM en Fashion-MNIST.
        Prepara los datos aplanados y en formato NumPy.
        """
        # Transformación EXCLUSIVA para KDM: Aplanar a 784 y dejar en [0, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x))
        ])

        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

        print("Cargando datasets Fashion-MNIST para KDM...")
        self.train_set_full = datasets.FashionMNIST(data_dir, train=True, download=True, transform=self.transform)
        self.test_set_full = datasets.FashionMNIST(data_dir, train=False, download=True, transform=self.transform)

    def get_all_numpy_data(self):
        """
        Extrae todo el dataset en formato NumPy para usar con Keras y Sklearn (K-Fold).
        """
        # Usamos un batch_size grande temporal solo para la extracción rápida a la RAM
        train_loader = DataLoader(self.train_set_full, batch_size=2000, shuffle=False)
        test_loader = DataLoader(self.test_set_full, batch_size=2000, shuffle=False)

        def to_numpy(loader):
            x_list, y_list = [], []
            for x, y in loader:
                x_list.append(x.numpy())
                y_list.append(y.numpy())
            return np.concatenate(x_list), np.concatenate(y_list)

        x_train, y_train = to_numpy(train_loader)
        x_test, y_test = to_numpy(test_loader)

        return x_train, y_train, x_test, y_test

    def get_test_loader(self):
        return DataLoader(self.test_set_full, batch_size=self.batch_size, shuffle=False)



# ==========================================
# MÓDULO MLOPS: dataset_loader.py (Adaptación SVHN)
# ==========================================



class SVHNDatasetLoader:
    """
    Clase para la gestión integral del dataset SVHN.
    Asegura la descarga, preprocesamiento y formateo tensorial
    para arquitecturas Kernel Density Matrix (KDM).
    """
    def __init__(self, data_dir="./data/svhn", flatten_input=True):
        self.data_dir = data_dir
        self.flatten_input = flatten_input
        self.urls = {
            "train": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "test": "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
        }

    def _verificar_y_descargar(self):
        """Método privado para asegurar la disponibilidad de los datos en local."""
        os.makedirs(self.data_dir, exist_ok=True)
        for split, url in self.urls.items():
            filepath = os.path.join(self.data_dir, f"{split}_32x32.mat")
            if not os.path.exists(filepath):
                print(f"[⬇️] Descargando partición {split} de SVHN...")
                urllib.request.urlretrieve(url, filepath)

    def load_data(self):
        """
        Carga y transforma los tensores.
        Retorna: X_train, y_train, X_test, y_test
        """
        self._verificar_y_descargar()
        print("[⚙️] Procesando y transponiendo tensores SVHN...")
        
        train_data = sio.loadmat(os.path.join(self.data_dir, "train_32x32.mat"))
        test_data = sio.loadmat(os.path.join(self.data_dir, "test_32x32.mat"))

        # El formato original de Stanford es (alto, ancho, canales, num_muestras)
        # Transponemos a (N, 32, 32, 3) estándar en MLOps con TensorFlow/Keras
        X_train = np.transpose(train_data['X'], (3, 0, 1, 2)).astype('float32') / 255.0
        X_test = np.transpose(test_data['X'], (3, 0, 1, 2)).astype('float32') / 255.0

        # Mapeo de etiquetas: el dígito '0' viene etiquetado como '10'
        y_train = train_data['y'].flatten()
        y_train[y_train == 10] = 0
        y_test = test_data['y'].flatten()
        y_test[y_test == 10] = 0

        # Aplanamiento de la matriz para KDMs (si no se usan capas convolucionales previas)
        if self.flatten_input:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

        return X_train, to_categorical(y_train, 10), X_test, to_categorical(y_test, 10)
