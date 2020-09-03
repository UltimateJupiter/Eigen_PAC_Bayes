import torch
import numpy as np
from utils import kp_2d, bkp_2d, bkp_2d_inplace
import sys

n = 3
A = np.array([[[0,1],[1,1]], [[1,1],[0,0]]])
B = np.arange(2*n*n).reshape(2,n,n)

a = torch.from_numpy(A)
b = torch.from_numpy(B)
print(a, b)

print(kp_2d(a[0], b[0]))
print(kp_2d(a[1], b[1]))
out = bkp_2d_inplace(a, b)
print(out)