import torch
from torch.nn import functional as F
import torch.nn as nn

from models.positional_encoding import LearnablePositionalEncoding


class QuestionGenerator(nn.Module):
    def __init__(self, feat_dim, pair_len, max_len, d_model, n_heads, num_layers, dim_feedforward,
                 dropout=0.1, activation='gelu'):
        super(QuestionGenerator, self).__init__()
        
        self.feat_dim = feat_dim
        self.pair_len = pair_len
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        
        self.project_input = nn.Sequential(
                             nn.Linear(feat_dim, d_model),
                             nn.Conv1d(in_channels=pair_len, out_channels=max_len, kernel_size=1)
                             )
        
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.activation = self._get_activation_fn(activation)
        self.dropout1 = nn.Dropout(dropout)
        
        self.output_layer = self.build_output_module(d_model, feat_dim)
        
    
    def build_output_module(self, d_model, feat_dim):
        output_layer = nn.Linear(d_model, feat_dim)
        return output_layer
    
    
    def freeze_parameters(self):
        for p in self.parameters():
            p.requires_grad = False # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    
    
    def restore_parameters(self):
        for p in self.parameters():
            p.requires_grad = True
        
    
    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise ValueError("activation should be relu/gelu, not {}".format(activation))
        
    
    def forward(self, x):
        x_in = self.project_input(x)
        x_in = x_in.permute(1, 0, 2)        
        x_in = self.pos_encoder(x_in)
        
        x_out = self.transformer_encoder(x_in)
        x_out = self.activation(x_out)
        x_out = x_out.permute(1, 0, 2)
        x_out = self.dropout1(x_out)
        x_out = self.output_layer(x_out)
        
        return x_out