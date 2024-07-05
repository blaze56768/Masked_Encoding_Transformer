__author__ = "Vikas Kumar"
__copyright__ = "Deutsche Telekom"

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch import nn,optim
import lightning as L
from .modules import positional_embedding,masking,transformer
import torch
import numpy as np
import wandb
class SSLModel(L.LightningModule):
    """This Module trains and test the ssl model

    Args:
        SSLModel (_cfg_): configs file
    """
    
    def __init__(self,cfg):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.cfg=cfg
  
        self.masker = masking.Masker(self.cfg)
        self.pos_emb = positional_embedding.PositionEmbedding(self.cfg)
        self.encoder = nn.ModuleList([transformer.TransformerBlock(self.cfg) for i in range(self.cfg.n_encoding)])   
        self.decoder = nn.ModuleList([transformer.TransformerBlock(self.cfg) for i in range(self.cfg.n_decoding)])
        
        self.proj_head_reg = nn.Linear(in_features=self.cfg.embed_dim+1, out_features=self.cfg.out_features)

        self.loss_mse = torch.nn.functional.mse_loss
        self.loss_bce = torch.nn.functional.binary_cross_entropy_with_logits
        
        return

    def forward(self,inputs,masking_pct):
        masked, masked_ind, unmasked, unmasked_ind = self.masker.forward(inputs,mask_pct=masking_pct)
        
        #get positional embeddings
        unmasked_pos_emb = self.pos_emb.forward(unmasked_ind)
        masked_pos_emb = self.pos_emb.forward(masked_ind)
        
        #concatenate unmasked tensor and unmasked positional embeddings
        unmasked = torch.unsqueeze(unmasked,-1)
        unmasked= unmasked.float()
        unmasked_emb = torch.cat([unmasked,unmasked_pos_emb],dim=2)
        
        #put unmasked embeddings into encoder
        for block in self.encoder:
            unmasked_emb = block(unmasked_emb)
        
        #concatenate masked tensor and masked positional embeddings
        masked = torch.unsqueeze(masked,-1)
        masked_emb = torch.concat((masked,masked_pos_emb),dim=2)
        
        #concatenate masked & unmasked embeddings
        all_emb = torch.concat((unmasked_emb,masked_emb),dim=1)
        all_emb = all_emb.float()
        
        #out embeddings into decoder
        for block in self.decoder:
            all_emb = block(all_emb)
        
        #get final outputs
        outputs = self.proj_head_reg(all_emb)
        outputs = torch.squeeze(outputs,dim=2)
        
        return outputs
        
    
    
    def training_step(self,batch,batch_idx):
        #numerical loss only version for the sample dataset
     
        inputs=batch[:,:-2]
        targets = batch[:,:-2]
       
        outputs = self(inputs,self.cfg.masking_pct)
        cur_loss = self.loss_mse(outputs,targets)
        logs = {"train/loss":cur_loss}
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
        
        return {"loss": cur_loss}
    
    def validation_step(self,batch,batch_idx):
       
        #numerical only case for  
        inputs=batch[:,:-1]
        targets = batch[:,:-1]
       
        outputs = self(inputs,self.cfg.masking_pct)
        cur_loss = self.loss_mse(outputs,targets)
    
        logs = {"val/loss":cur_loss}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        return {"loss": cur_loss}
    
    def test_step(self,batch,batch_idx):
        
        inputs=batch[:,:-1]
        targets = batch[:,:-1]
       
        outputs = self(inputs)
        cur_loss = self.loss_mse(outputs,targets,self.cfg.masking_pct)
        #loss_num,loss_cat = self.reconstruction_loss(outputs,targets)
        logs = {"test/loss":cur_loss}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        
        return {"loss": cur_loss}
    

    def reconstruction_loss(self,outputs,targets):
        outputs_categorical, outputs_numerical = self.split(outputs)
        targets_categorical, targets_numerical = self.split(targets)

        loss_num = self.loss_mse(outputs_numerical,targets_numerical)
        loss_cat = self.loss_bce(outputs_categorical,targets_categorical)
        return loss_num,loss_cat
        
    def split(self, x):
        return torch.split(x, [self.cfg.features_categorical, self.cfg.features_numerical], dim=1)
    
    def on_train_start(self):  
        print("✅Training is about to start")  
  
    def on_train_end(self):  
        print("🎉Training has ended🎉")  
        
    def on_test_start(self):  
        print("✅Testing is about to start")  
  
    def on_test_end(self):  
        print("🎉Testing has ended🎉")

    def configure_optimizers(self):
        self.optimizer = optim.Adam(params=self.parameters(),lr=self.cfg.lr,amsgrad=True)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.cfg.lr_gamma) 
        self.scheduler= optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.lr_step_size)
        return [self.optimizer],[self.scheduler]

        
    
    
  