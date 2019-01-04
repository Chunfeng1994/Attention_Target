

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self,config):
        super(Attention, self).__init__()
        self.config = config
        self.linear_1 = nn.Linear(config.hidden_dim * 4, config.attention_size, bias=True)
        self.u = nn.Linear(config.attention_size, 1, bias=False)


    def forward(self, hi,ht):
        ht = torch.mean(ht,1)
        ht = torch.unsqueeze(ht, 1)
        ht = ht.repeat(1, hi.size(1), 1)
        h = torch.cat((hi,ht),2)

        h = self.linear_1(h)
        h = F.tanh(h)
        u = self.u(h).squeeze(2)
        #bi结束
        h = F.softmax(u,dim=1)
        h = torch.unsqueeze(h, 2)
        h = h.repeat(1, 1, hi.size(2))
        #a结束
        s = torch.mul(h,hi)
        s = torch.sum(s, 1)
        return s



