import torch
import torch.nn as nn
from model.Bilstm import *
from model.Attention import *
class Contextualized(nn.Module):
    def __init__(self, config, embed_size, padding_idx, label_size):
        super(Contextualized, self).__init__()
        self.bilstm = Bilstm(config, embed_size)

        self.attention_s = Attention(config)
        self.attention_l = Attention(config)
        self.attention_r = Attention(config)

        self.linear_out = nn.Linear(config.hidden_dim * 6, label_size)

    def forward(self, w, start, end,length):
        s_part,target_part,left_part,right_part  = self.bilstm(w, start, end,length)

        s = self.attention_s(s_part,target_part)
        sl = self.attention_l(s_part, target_part)
        sr = self.attention_r(s_part, target_part)

        s_cat = torch.cat([s,sl,sr],1)

        s_final = self.linear_out(s_cat)

        return s_final