import torch 
import torch.nn as nn
import torch.nn.functional as F 

from  torch.nn.utils.rnn import pack_padded_sequence
# pack_padded_sequence: deal with padded_sequece


a = torch.tensor([[1,2,3,0,0],
                  [1,2,0,0,0], 
                  [1,0,0,0,0]])
a_len = torch.tensor([3,2,1])

print(a, a_len)

packed = pack_padded_sequence(a, lengths=a_len, batch_first=True)
print(type(packed))

print(packed.data)
print(packed.__dir__())
print(packed.batch_sizes)
