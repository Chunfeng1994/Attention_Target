import torch

x = torch.randn(1,4,3)
y = torch.randn(2,4,3)
s = x.contiguous().view(4,3)

s_index = [0,2]

hi_part = torch.index_select(s, 0, torch.LongTensor(s_index))
hl_part = hi_part.view(1, len(s_index),3)

a = [[1,2],[2,3]]
b = []
for i in range(len(a)):
   print(a[i])
