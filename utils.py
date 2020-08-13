import torch
import datetime
from queue import PriorityQueue
import pynvml
import numpy as np
import time
import itertools


def prepare_net(net, use_gpu=True):

    handle = None
    device = 'cpu'
    if not use_gpu:
        print('Running on CPUs')
        return net, device, handle
    
    if torch.cuda.is_available():
        device = 'cuda'

    if device != 'cpu':
        import pynvml
        import torch.backends.cudnn as cudnn

        print('Running on GPU')
        net = net.to(device)
        # net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        device_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        print("CUDA Device: {} | RAM: {:.4g}G".format(device_name, mem_info.total/(2**30)))
    else:
        print('No CUDA devices available, run on CPUs')
    
    return net, device, handle
