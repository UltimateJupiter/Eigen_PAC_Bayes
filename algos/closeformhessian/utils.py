import torch
import datetime
from queue import PriorityQueue
import pynvml
import numpy as np
import time
import itertools

def log(info, print_log=True):
    if print_log:
        print("[{}]  {}".format(datetime.datetime.now(), info))

def timer(st, info='', timer_on=True):
    if timer_on:
        print("[{}]  {} {}".format(datetime.datetime.now(), datetime.datetime.now() - st, info))

def est(st, progress, info=''):
    time_diff = datetime.datetime.now() - st
    est = (1 - progress) * time_diff / progress
    print("[{}]  {:.3g}%\t EST: {}".format(datetime.datetime.now(), progress * 100, str(est)[:-4]))

def get_time_seed():
    return int(str(datetime.datetime.now())[-4:])

def get_tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()

def get_tensor_dict_size(tensor_dict, p=False):
    ret = 0
    for k in tensor_dict:
        ret += get_tensor_size(tensor_dict[k])
    if p:
        print("{:.4g} G".format(ret / (2 ** 30)))
    return ret

def gpu_memory(print_out=True):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        time.sleep(0.2)
        if print_out:
            log("{} {:.5g}% {:.4g}G".format(pynvml.nvmlDeviceGetName(handle).decode("utf-8"), 100 * mem_info.used / mem_info.total, mem_info.total/(2**30)))
        return mem_info.used, mem_info.total
    except:
        return None, None

def empty_cache(device):
    if device == 'cpu':
        return
    else:
        torch.cuda.empty_cache()

def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device) # pylint: disable=no-member
    indices = torch.arange(result.numel(), device=device).reshape(shape) # pylint: disable=no-member
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

def kmax_argsort(a, b, k, return_vals=False):

    q = PriorityQueue()
    la, lb = len(a), len(b)
    assert k <= la * lb
    
    q.put((- a[0] * b[0], (0, 0)))
    vals, args = [], []
    args_set = set((0, 0))

    for _ in range(k):
    
        val, (ia, ib) = q.get()
        vals.append(-val)
        args.append((ia, ib))
    
        if ia + 1 < la:
            if (ia + 1, ib) not in args_set:
                args_set.add((ia + 1, ib))
                q.put((- a[ia + 1] * b[ib], (ia + 1, ib)))
    
        if ib + 1 < lb:
            if (ia, ib + 1) not in args_set:
                args_set.add((ia, ib + 1))
                q.put((- a[ia] * b[ib + 1], (ia, ib + 1)))
    
    if return_vals:
        return args, vals
    else:
        return args

def kp_2d(t1, t2):
    t1_h, t1_w = t1.size()
    t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    tiled_t2 = t2.repeat(t1_h, t1_w)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_h, t2_w, 1)
          .view(out_h, out_w)
    )
    return expanded_t1 * tiled_t2

def bkp_2d_raw(t1, t2):
    btsize, t1_h, t1_w = t1.size()
    btsize, t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    tiled_t2 = t2.repeat(1, t1_h, t1_w)
    expanded_t1 = (
        t1.unsqueeze(3)
          .unsqueeze(4)
          .repeat(1, 1, t2_h, t2_w, 1)
          .view(btsize, out_h, out_w)
    )
    expanded_t1 *= tiled_t2
    return expanded_t1

def bkp_2d(t1, t2):
    btsize, t1_h, t1_w = t1.size()
    btsize, t2_h, t2_w = t2.size()
    out_h = t1_h * t2_h
    out_w = t1_w * t2_w

    expanded_t1 = (
        t1.unsqueeze(3)
          .unsqueeze(4)
          .repeat(1, 1, t2_h, t2_w, 1)
          .view(btsize, out_h, out_w)
    )
    for i in range(t1_h):
        for j in range(t1_w):
            expanded_t1[:, i * t2_h: (i + 1) * t2_h, j * t2_w: (j + 1) * t2_w] *= t2
    return expanded_t1



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

def eigenthings_tensor_utils(t, device=None, out_device='cpu', symmetric=False, topn=-1):
    t = t.to(device)
    if topn >= 0:
        _, eigenvals, eigenvecs = torch.svd_lowrank(t, q=min(topn, t.size()[0], t.size()[1]))
        eigenvecs.transpose_(0, 1)
    else:
        if symmetric:
            eigenvals, eigenvecs = torch.symeig(t, eigenvectors=True) # pylint: disable=no-member
            eigenvals = eigenvals.flip(0)
            eigenvecs = eigenvecs.transpose(0, 1).flip(0)
        else:
            _, eigenvals, eigenvecs = torch.svd(t, compute_uv=True) # pylint: disable=no-member
            eigenvecs = eigenvecs.transpose(0, 1)
    return eigenvals, eigenvecs

def eigenthings_tensor_utils_batch(t, device=None, out_device='cpu', symmetric=False, topn=-1):
    t = t.to(device)
    assert len(t.shape) == 3
    vals, vecs = [], []
    for t_sample in t:
        eigenvals, eigenvecs = eigenthings_tensor_utils(t_sample, device=device, out_device=out_device, symmetric=symmetric, topn=topn)
        vals.append(eigenvals.unsqueeze(0))
        vecs.append(eigenvecs.unsqueeze(0))
    vals = torch.cat(vals) # pylint: disable=no-member
    vecs = torch.cat(vecs) # pylint: disable=no-member
    return vals, vecs

class Utils():

    def __init__(self, HM, device):
        self.device = device
        self.HM = HM
        return

    
    def increasing_interval_dist(self, s, t, d=1, r=1.1):
        ret = [s]
        while True:
            if ret[-1] + d >= t:
                break
            ret.append(ret[-1] + d)
            d *= r
        return np.array(ret, dtype=int)
    
    def reshape_to_layer(self, tensor, layer, out_device=None):
        ret = tensor.view_as(self.HM.sd[layer + '.weight'])
        if out_device == 'cpu':
            ret = ret.cpu()
        else:
            ret.to(self.HM.device)
        return ret
    
    def mm(self, m1, m2):
        device_inp = m1.device
        m1 = m1.to(self.device)
        m2 = m2.to(self.device)
        return torch.mm(m1, m2).to(device_inp) # pylint: disable=no-member

    def timer(self, st, info=''):
        print("[{}]  {} {}".format(datetime.datetime.now(), datetime.datetime.now() - st, info))
    
    def log(self, info, print_log=True):
        if print_log:
            print("[{}]  {}".format(datetime.datetime.now(), info))

    def now(self):
        return datetime.datetime.now()

    def eigenthings_tensor_utils(self, t, device=None, out_device='cpu', symmetric=False, topn=-1):
        """return eigenvals, eigenvecs"""
        t = t.to(device)
        if topn >= 0:
            _, eigenvals, eigenvecs = torch.svd_lowrank(t, q=min(topn, t.size()[0], t.size()[1]))
            eigenvecs = eigenvecs.transpose(0, 1)
        else:
            if symmetric:
                eigenvals, eigenvecs = torch.symeig(t, eigenvectors=True) # pylint: disable=no-member
                eigenvals = eigenvals.flip(0)
                eigenvecs = eigenvecs.transpose(0, 1).flip(0)
            else:
                _, eigenvals, eigenvecs = torch.svd(t, compute_uv=True) # pylint: disable=no-member
                eigenvecs = eigenvecs.transpose(0, 1)
        return eigenvals, eigenvecs
    
    def gram_schmidt(self, vv):

        vv = vv.to(self.device)
        def projection(u, v):
            return v.dot(u) / u.dot(u) * u

        nk = vv.shape[0]
        uu = torch.zeros_like(vv, device=vv.device) # pylint: disable=no-member
        uu[0] = vv[0].clone()
        for k in range(1, nk):
            vk = vv[k]
            uk = 0
            for j in range(0, k):
                uj = uu[j]
                uk = uk + projection(uj, vk)
            uu[k] = vk - uk
        for k in range(nk):
            uk = uu[k]
            uu[k] = uk / uk.norm()
        return uu
