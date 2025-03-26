import torch
import numpy as np
from torch import nn as nn
b=[]

a=torch.tensor([1,2,-1,2,3,4])
a=torch.randn(5,5)

b=torch.tensor([[1,2,3,4]])
print(a)
print(b)
print(a[b])
