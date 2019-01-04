import torch
import torch.nn as nn
from model.Bilstm import *
from model.Attention import *
class ContextualizedGates(nn.Module):
    def __init__(self, config, embed_size, padding_idx, label_size):
        super(ContextualizedGates, self).__init__()
        self.bilstm = Bilstm(config, embed_size)

        self.attention_s = Attention(config)
        self.attention_l = Attention(config)
        self.attention_r = Attention(config)
        self.u1 = nn.Linear(config.hidden_dim * 4,config.hidden_dim * 2)
        self.u2 = nn.Linear(config.hidden_dim * 4,config.hidden_dim * 2)
        self.u3 = nn.Linear(config.hidden_dim * 4,config.hidden_dim * 2)
        self.linear_out = nn.Linear(config.hidden_dim * 2, label_size)

    def forward(self, w, start, end,length):
        s_part,target_part,left_part,right_part = self.bilstm(w, start, end,length)

        s = self.attention_s(s_part,target_part)
        sl = self.attention_l(s_part, target_part)
        sr = self.attention_r(s_part, target_part)

        ht = torch.mean(target_part,1)

        z = self.u1(torch.cat([s, ht], 1))
        zl = self.u2(torch.cat([sl, ht], 1))
        zr = self.u3(torch.cat([sr, ht], 1))

        s_out = torch.mul(z,s) + torch.mul(zl,sl) + torch.mul(zr,sr)



        s_final = self.linear_out(s_out)

        return s_final