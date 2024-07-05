__author__ = "Luis Felipe Villa Arenas"
__copyright__ = "Deutsche Telekom"

import lightning as L
from torch import nn,relu

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.multihead_attention = nn.MultiheadAttention(self.cfg.embed_dim+1,
                                                               self.cfg.n_heads,
                                                               self.cfg.dropout,
                                                               batch_first=True)
        
        self.feedforward = PositionwiseFeedForward(self.cfg.embed_dim+1, 
                                                   self.cfg.d_feedforward,
                                                   self.cfg.dropout)

        self.dropout= nn.Dropout(self.cfg.dropout)
        self.layernorm_attention = nn.LayerNorm(self.cfg.embed_dim+1)
        self.layernorm_feedforward = nn.LayerNorm(self.cfg.embed_dim+1)

        
        return
    
    def forward(self,x):
        # Multi-Head Attention.
        attention_out, _ = self.multihead_attention(x, x, x)
        # Add & Norm
        x = self.layernorm_attention(x + self.dropout(attention_out))
        # Position wise FeedForward
        feedforward_out = self.feedforward(x)
        # Add & Norm
        x = self.layernorm_feedforward(x + self.dropout(feedforward_out))
        return x

    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, embed_dim, feedforward_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_in = nn.Linear(embed_dim, feedforward_dim)
        self.linear_out = nn.Linear(feedforward_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear_out(self.dropout(relu(self.linear_in(x))))
