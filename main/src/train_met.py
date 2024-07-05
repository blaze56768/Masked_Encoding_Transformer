import os
import hydra
from omegaconf import DictConfig
from datamodules import dataset
from models import ssl
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from utils import utils_funcs
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor,ModelCheckpoint

@hydra.main(config_path = "../configs/sample/scaled/pretrain",config_name ="msk_60.yaml")  
def train(cfg : DictConfig)->None:
    wandb_logger = WandbLogger(project=cfg.pretrain_wandb,group="toy",job_type = "train")
    
    print("✅Initializing the data loader...")
    sslloader = dataset.SSLDataModule(cfg=cfg)   
    
    sslloader.setup("fit")
    sslloader.setup("val")
    #cfg.lr_step_size = (int(sslloader.dataset_train.data.shape[0]/cfg.batch_size))*cfg.n_epochs
    
    
    train_loader =sslloader.train_dataloader()
    val_dataloader =sslloader.val_dataloader()
   
       
    print("✅Setup phase")  
    model = ssl.SSLModel(cfg=cfg)    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.pretrain_checkpoint, save_top_k=1, monitor="train/loss",mode="min",filename="msk_60")
    trainer = L.Trainer(max_epochs=cfg.n_epochs,logger = wandb_logger,check_val_every_n_epoch=cfg.val_epochs,callbacks=[checkpoint_callback,lr_monitor])
    
    wandb_logger.watch(model,log="all")
    trainer.fit(model=model,train_dataloaders=train_loader,val_dataloaders=val_dataloader)
    #trainer.save_checkpoint(cfg.pretrain_checkpoint)  
    checkpoint_callback.best_model_path
    
    wandb.finish()
    return cfg

if __name__ == "__main__":
    train()
        