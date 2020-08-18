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

def log(info, print_log=True):
    if print_log:
        print("[{}]  {}".format(datetime.datetime.now(), info))

def get_tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()

def get_tensor_dict_size(tensor_dict, p=False):
    ret = 0
    for k in tensor_dict:
        ret += get_tensor_size(tensor_dict[k])
    if p:
        print("{:.4g} G".format(ret / (2 ** 30)))
    return ret

def network_params(model):
    """
    Return a list containing names and shapes of neural network layers
    """

    layers = []
    ind = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = model.state_dict()[name].shape
            params = np.ravel(param.data.cpu().numpy())
            ind2 = np.size(params)
            ind = ind2
            layers.append((name, ind, shape))
        
    return layers