from abc import ABC, abstractmethod
class BaseDataPipeline(ABC):
    def __init__(self, data_dir='./data', batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size
    @abstractmethod
    def load_data(self): pass
    @abstractmethod
    def get_kdm_data(self): pass
    @abstractmethod
    def get_pnc_loaders(self): pass
