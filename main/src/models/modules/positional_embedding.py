__author__ = "Hyemin Kim"
__copyright__ = "Deutsche Telekom"

from torch import nn
import lightning as L
import torch

class PositionEmbedding(torch.nn.Module):
    """This Modules computes positional embedding of both masked & unmasked positions, generates unmasked array that contains positional embeddings.

    Args:
        PositionEmbedding (_cfg_): configs file
    """
    
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.d_model = self.cfg.embed_dim
        self.pos_emb = nn.Embedding(num_embeddings=self.cfg.d_feature, embedding_dim=self.cfg.embed_dim)
      
        
    def forward(self,ind):
        """This method generates positional embeddings

        Args:
            unmasked (_torch tensor_): unmasked elements
            masked_ind (_torch tensor_): indices of masked elements
            unmasked_ind (_torch tensor_): indices of unmasked elements

        Returns:
            unmasked: unmasked elements concatenated with embeddied unmasked indices
            unmasked_ind: indices lists of unmasked elements
            masked_ind: embedded indices of masked elements
        """
       
        #embedd unmasked indices
        pos_emb= self.pos_emb(ind)
        return pos_emb
        
        
 
        
        
        
        
        