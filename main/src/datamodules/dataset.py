__author__ = "Vikas Kumar"
__copyright__ = "Deutsche Telekom"

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Dataset
import pandas as pd
import torch
import lightning as L
from torch.utils.data import  DataLoader

class SSLDataset(Dataset):
    """_This Class loads input data and return data given their indices_

    Args:
        path (string): path of data file
        
    """
    def __init__(self,path,data=None):
        super().__init__()
        self.path = path
        self.data = pd.read_csv(path)
       
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) :
        arr=self.data.loc[index]
        #input = arr[:,:-2]
        #labels =arr[:,-1]
        #return torch.Tensor(input),torch.Tensor(labels)
        return torch.Tensor(arr)

class SSLDataModule(L.LightningDataModule):
    """_This Data Module defines the dataset and data loader, depending on the stage_

    Args:
        cfg (_cfg_): configs file
    """
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
       
        
    def setup(self, stage="fit"):
        if stage == "fit":
            self.dataset_train = SSLDataset(path=self.cfg.train_pth)
        elif stage =="test":
            self.dataset_test = SSLDataset(path=self.cfg.test_pth)
        else:
            self.dataset_test = SSLDataset(path=self.cfg.test_pth)
            
    def train_dataloader(self):
        return DataLoader(self.dataset_train,batch_size=self.cfg.batch_size,shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.dataset_test,batch_size=self.cfg.batch_size,shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_test,batch_size=self.cfg.batch_size,shuffle=True)
        
        
        
    

        
