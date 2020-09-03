import torch
import numpy as np
from .utils import *

class Measures():

    def __init__(self, device):
        self.device = device
        return
    
    def trace_overlap(self, s1, s2, device=None):
        if device is None:
            device = self.device
        M = s1.to(device)
        V = s2.to(device)
        Mt = torch.transpose(M, 0, 1) # pylint: disable=no-member
        n = M.size()[0]
        k = 0
        for i in range(n):
            vi = V[i]
            li = Mt.mv(M.mv(vi))
            ki = torch.dot(li, li) # pylint: disable=no-member
            k += ki
        del Mt, M, V
        torch.cuda.empty_cache()
        return float((k / n).cpu().numpy())
    
    def trace_overlap_trend(self, s1, s2, ls, device=None):
        ret = []
        for dim in ls:
            dim = int(dim)
            assert dim > 0
            assert dim <= min(s1.size()[0], s2.size()[0]), (s1.size(), s2.size())
            ret.append(self.trace_overlap(s1[:dim], s2[:dim], device=None))
        return np.array(ret)