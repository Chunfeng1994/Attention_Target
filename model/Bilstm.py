import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.init as init

class Bilstm(nn.Module):
    def __init__(self,config,embed_size,embedding):

        super(Bilstm, self).__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.embed_dim = config.embed_dim
        self.dropout = nn.Dropout(config.dropout)


        self.embed = nn.Embedding(embed_size, config.embed_dim, max_norm=config.max_norm, scale_grad_by_freq=True)
        if embedding is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embedding))
        self.bilstm = nn.LSTM(config.embed_dim, config.hidden_dim, num_layers=config.num_layers, bias=True, bidirectional=True)

    def forward(self, x, start, end, length):

        x = self.embed(x)
        x = self.dropout(x)


        h = pack_padded_sequence(x, length)
        h, _ = self.bilstm(h)
        h, _ = pad_packed_sequence(h)


        h = h .transpose(0,1)
        x_size = h.size(0)
        y_size = h.size(1)
        z_size = h.size(2)
        h_tem = h.contiguous().view(x_size * y_size, z_size)

#扩充padding补 0
        h_tem = torch.cat([h_tem,torch.zeros(1, z_size)], 0)
        PAD = x_size * y_size

        s = []
        target = []
        left = []
        right = []
        target_index = []
        s_index = []
        left_index = []
        right_index = []
        hl_part = None
        hr_part = None



        # # batch = 1
        #
        # for index in range(length[0]):
        #     s_index.append(index)
        #
        # for index in range(end[0] + 1):
        #     left_index.append(index)
        #
        #
        # for index in range(start[0],length[0]):
        #     right_index.append(index)
        #
        # for index in range(start[0], end[0] + 1):
        #     target_index.append(index)
        #     s_index.remove(index)
        #     left_index.remove(index)
        #     right_index.remove(index)

        # ht_part = torch.index_select(h_tem, 0, torch.LongTensor(target_index))
        # ht_part = ht_part.view(x_size, len(target_index), z_size)
        #
        # hi_part = torch.index_select(h_tem, 0, torch.LongTensor(s_index))
        # hi_part = hi_part.view(x_size, len(s_index), z_size)
        #
        # if len(left_index) != 0:
        #     hl_part = torch.index_select(h_tem, 0, torch.LongTensor(left_index))
        #     hl_part = hl_part.view(x_size, len(left_index), z_size)
        # if len(right_index) != 0:
        #     hr_part = torch.index_select(h_tem, 0, torch.LongTensor(right_index))
        #     hr_part = hr_part.view(x_size, len(right_index), z_size)

 #batch > 1

        sentence_length_max = max(length)

        target_length_max = max([e - s + 1 for (s,e) in zip(start,end)])

        left_length_max = max(start)

        right_length_max = max([l - e - 1 for (e,l) in zip(end,length)])

        sentence_not_target_length_max = max(length[i] - (e - l + 1) for i,(l, e) in enumerate(zip(start, end)))  # 去掉target之后的一句话最大长度


        #target
        for i in range(len(length)):
            l = []
            for j in  range(start[i],end[i] + 1):
                l.append(j + sentence_length_max * i)
            k = end[i] - start[i] + 1
            if k < target_length_max:
                for j in range(target_length_max - k):
                    l.append(PAD)
            target.append(l)
            target_index += l

        #left
        for i in range(len(length)):
            l = []
            for j in range(start[i]):
                l.append(j + sentence_length_max * i)
            k = start[i]
            if k < left_length_max:
                for j in range(left_length_max - k):
                    l.append(PAD)
            left.append(l)
            left_index += l

        #right
        for i in range(len(length)):
            l = []
            for j in range(end[i] + 1,length[i]):
                l.append(j + sentence_length_max * i)
            k = length[i] - end[i] - 1
            if k < right_length_max:
                for j in range(right_length_max - k):
                    l.append(PAD)
            right.append(l)
            right_index += l

        #sentence
        for i,s_length in enumerate(length):
            l = []
            for j in range(s_length):
                if start[i] <= j <= end[i]:
                    continue
                else:
                    l.append(j + sentence_length_max * i)
            k = len(l)
            if k < sentence_not_target_length_max:
                for j in range(sentence_not_target_length_max - k):
                    l.append(PAD)
            s.append(l)
            s_index += l

       



        ht_part = torch.index_select(h_tem, 0, torch.LongTensor(target_index))
        ht_part = ht_part.view(x_size,target_length_max,z_size)


        hi_part = torch.index_select(h_tem, 0, torch.LongTensor(s_index))
        hi_part = hi_part.view(x_size, sentence_not_target_length_max, z_size)

        if len(left_index) != 0:
            hl_part = torch.index_select(h_tem, 0, torch.LongTensor(left_index))
            hl_part = hl_part.view(x_size, left_length_max, z_size)
        if len(right_index) != 0:
            hr_part = torch.index_select(h_tem, 0, torch.LongTensor(right_index))
            hr_part = hr_part.view(x_size, right_length_max, z_size)




        return hi_part,ht_part,hl_part,hr_part









