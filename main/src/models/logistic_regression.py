"""
src.models.transformer_classifier.py

This module uses a pretrain encoder combined with MLP to classify abnormalities.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

__author__ = "Vikas Kumar"
__copyright__ = "Deutsche Telekom"


import lightning as L

import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
from models import ssl

class DownstreamClassifier(L.LightningModule):
    """
    Downstream Classifier.
    """
    def __init__(self,cfg):
        
        super().__init__()

        # Hyperparameters
        self.save_hyperparameters()

        self.cfg = cfg

        if self.cfg.downstream_pretrain:
            # Init the pretrained LightningModule
            self.pretrained=ssl.SSLModel.load_from_checkpoint(self.cfg.pretrain_checkpoint)

            # Freeze Pretrain encoder       
            self.pretrained.freeze()

        # Init Layers
        self.projection_layer = nn.Linear(self.cfg.d_feature,self.cfg.out_features)

        # Criterion
        self.criterion = nn.BCELoss()

    def forward(self, x):
        # Feature extractor

        if self.cfg.downstream_pretrain:
            embeddings = self.pretrained(x,0)
        else:
            embeddings = x
        
        pred= self.projection_layer(embeddings)
        pred = torch.sigmoid(pred)
        pred=torch.squeeze(pred,-1)
        
        return pred

    def training_step(self, batch, batch_idx):
        
        input = batch[:,:-2]
        labels =batch[:,-1]
        predictions = self(input)
        loss = self.criterion(predictions, labels)
        
        acc= torchmetrics.functional.accuracy(predictions,labels.int(),task="binary")
        prec= torchmetrics.functional.precision(predictions,labels,task="binary")
        recall= torchmetrics.functional.recall(predictions,labels.int(),task="binary")
        f1= torchmetrics.functional.f1_score(predictions, labels.int(),task="binary")
        auroc = torchmetrics.AUROC(task="binary")
        auc = auroc(predictions,labels.int())
        apre = torchmetrics.AveragePrecision(task="binary")
        ap = apre(predictions,labels.int())
        
        logs={"train/loss":loss,"train/accuracy":acc,
              "train/precision":prec,"train/recall":recall, 
              "train/f1":f1,"train/auc":auc,
              "train/ap":ap}
        self.log_dict(
            logs,
            on_step=True, on_epoch=False, prog_bar=True, logger=True
        )
       
        return {"loss":loss}

   

    def validation_step(self, batch, batch_idx):
        input = batch[:,:-1]
        labels =batch[:,-1]
        predictions = self(input)
        loss = self.criterion(predictions, labels)
        
        acc= torchmetrics.functional.accuracy(predictions,labels.int(),task="binary")
        prec= torchmetrics.functional.precision(predictions,labels,task="binary")
        recall= torchmetrics.functional.recall(predictions,labels.int(),task="binary")
        f1= torchmetrics.functional.f1_score(predictions, labels.int(),task="binary")
        auroc = torchmetrics.AUROC(task="binary")
        auc = auroc(predictions,labels.int())
        apre = torchmetrics.AveragePrecision(task="binary")
        ap = apre(predictions,labels.int())
        
        logs={"val/loss":loss,"val/accuracy":acc,
              "val/precision":prec,"val/recall":recall,
              "val/f1":f1, "val/auc":auc,
              "val/ap":ap
              }
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
       
        return {"loss":loss}


    def test_step(self, batch, batch_idx):
        input = batch[:,:-1]
        labels =batch[:,-1]
        predictions = self(input)
        loss = self.criterion(predictions, labels)
        
        acc= torchmetrics.functional.accuracy(predictions,labels.int(),task="binary")
        prec= torchmetrics.functional.precision(predictions,labels,task="binary")
        recall= torchmetrics.functional.recall(predictions,labels.int(),task="binary")
        f1= torchmetrics.functional.f1_score(predictions, labels.int(),task="binary")
        auroc = torchmetrics.AUROC(task="binary")
        auc = auroc(predictions,labels.int())
        apre = torchmetrics.AveragePrecision(task="binary")
        ap = apre(predictions,labels.int())
        
        logs={"test/loss":loss,"test/accuracy":acc,
              "test/precision":prec,"test/recall":recall, 
              "test/f1":f1,"test/auc":auc,
              "test/ap":ap}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"loss":loss}
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(params=self.parameters(),lr=self.cfg.downstream_lr,amsgrad=True) 
        #self.scheduler= torch.optim.lr_scheduler.StepLR(self.optimizer, self.cfg.downstream_lr_step_size)
        self.scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.downstream_lr_step_size)
        return [self.optimizer],[self.scheduler]

  