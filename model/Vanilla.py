
import torch.nn as nn
import torch.nn.init as init
from model.Bilstm import *
from model.Attention import *
class Vanilla(nn.Module):
    def __init__(self, config, embed_size, padding_idx, label_size,embedding):

        super(Vanilla, self).__init__()
        self.bilstm = Bilstm(config,embed_size,embedding)

        self.attention = Attention(config)

        self.linear_out = nn.Linear(config.hidden_dim * 2, label_size)



    def forward(self, w, start, end,length):
        s_word,t_word,l_word,r_word= self.bilstm(w,start,end,length)

        s = self.attention(s_word,t_word)
        s = self.linear_out(s)
        return s