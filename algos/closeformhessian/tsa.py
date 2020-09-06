import torch
import scipy.linalg as sla
import numpy as np
from inspect import signature
from copy import deepcopy
import itertools
import scipy.linalg as sla
import itertools
import datetime
from .utils import *
from .dp import *

def matseq_mul_t(mats, transposes):

    assert len(mats) == len(transposes)
    ret = None # type: torch.Tensor
    for i, m in enumerate(mats):
        mat = m.clone()
        if transposes[i]:
            mat.transpose_(-1, -2)
        if ret is None:
            ret = mat
        else:
            ret = torch.mm(ret, mat) if len(ret.shape) == 2 else torch.bmm(ret, mat) # pylint: disable=no-member
    return ret

def auto_batchsize_output(device, sample, max_cap=1024, RAM_cap=32):

    RAM_cap *= (2 ** 30)
    if device == 'cpu':
        if isinstance(sample, dict):
            sample_size = get_tensor_dict_size(sample)
        else:
            sample_size = get_tensor_size(sample)
        valid_size = int(RAM_cap / sample_size / 1.2) - 1
        # print("CPU RAM Cap Designated: {:.4g}".format(self.RAM_cap / (2**30)))
    else:
        if isinstance(sample, dict):
            sample_size = get_tensor_dict_size(sample)
        else:
            sample_size = get_tensor_size(sample)
        used_mem, total_mem = gpu_memory()
        valid_size = int((total_mem - used_mem) / sample_size / 1.2) - 1
    del sample
    empty_cache(device)
    if valid_size < 1:
        print("Memory Limit Too Small. Need at least {:.4g}G".format((sample_size * 1.2 + 1) / (2**30)))
        exit()
    log("batchsize predicted on output: {}".format(valid_size))
    return min(valid_size, max_cap)
    
class chain_comp_base():

    def __init__(self, funcs, transposes=None):

        # if not isinstance(funcs, list):
        if not hasattr(funcs, '__len__'):
            funcs = [funcs]
        self.funcs = funcs

        if transposes is None:
            transposes = [False for x in funcs]
        transposes = [bool(x) for x in transposes]
        assert len(funcs) == len(transposes), (len(funcs), len(transposes))
        
        self.transposes = transposes
        self.func_names = []
        for i, f in enumerate(funcs):
            f_name = f.__name__ # type:str
            assert f_name.endswith('_comp'), f_name
            f_name = f_name[:-5]
            if transposes[i]:
                f_name += 'T'

            self.func_names.append(f_name)

class normalize_comp_():

    def __init__(self, func, dim=-1):
        """dim: 0:col, 1:row, -1:fro, -2:trace, default:-1"""
        self.__name__ = func.__name__[:-5]
        self.dim = dim
        self.norm_dim = None
        if dim == -1:
            self.norm_dim = [1, 2]
            self.__name__ += 'nf'
        elif dim == 0:
            self.norm_dim = [1]
            self.__name__ += 'nc'
        elif dim == 1:
            self.norm_dim = [2]
            self.__name__ += 'nr'
        elif dim == -2:
            self.__name__ += 'nt'
        else:
            print('invalid normalization mode')
            exit()
        self.func = func

        self.__name__ += '_comp'
        log('{} initialized'.format(self.__name__))
    
    def __call__(self, ife, layers, inputs, device, Ws, out_device, batch_sum=False, **kwargs):
        out = self.func(ife, layers, inputs, device, Ws, out_device, batch_sum=False, **kwargs)
        for layer in out.keys():
            if self.dim == -2:
                # batched t, **kwargsrace
                norm_mat = torch.einsum('bii->b', out[layer]).unsqueeze(-1).unsqueeze(-1)
            else:
                norm_mat = out[layer].norm(dim=self.dim, keepdim=True)
            
            # avoid normalizing all zero matrices
            zero_entries = (norm_mat == 0.0)
            # print(zero_entries.sum())
            norm_mat += zero_entries
            out[layer].div_(norm_mat)

            if batch_sum:
                out[layer] = out[layer].sum(axis=0)
        return out

class chain_comp_(chain_comp_base):

    def __init__(self, funcs, transposes=None):
        super().__init__(funcs, transposes)
        self.name = '.'.join(self.func_names) + '_comp'
        self.__name__ = self.name

        log('{} initialized'.format(self.name))

    def __call__(self, ife, layers, inputs, device, Ws, out_device, batch_sum=False, **kwargs):
        mats_d = []

        for i, f in enumerate(self.funcs):
            j = self.funcs.index(f)
            mat_d = f(ife, layers, inputs, device, Ws, out_device, batch_sum=False, **kwargs) if i == j else mats_d[j]
            mats_d.append(mat_d)

        ret = {}
        for layer in layers:
            mats_layer = [mats_d[i][layer] for i in range(len(mats_d))]
            ret[layer] = matseq_mul_t(mats_layer, self.transposes)
            if batch_sum:
                ret[layer] = ret[layer].sum(axis=0)
        return ret

class stats_base():

    def __init__(self, HM, comp_layers, out_device):

        self.HM = HM
        self.comp_layers = comp_layers

        self.ds = HM.dataset
        self.dl = HM.dl
        self.out_device = out_device
        self.cache = HM.cache
    
    def sample_out(self, func, sample_count=1, batch_sum=False):

        sample_data = sample_input(self.ds, sample_count, remain_labels=self.HM.remain_labels)
        inputs, labels = sample_data[0].to(self.HM.device), sample_data[1].to(self.HM.device)
        return func(self.HM.ife, self.comp_layers, inputs, self.HM.device, self.HM.Ws, out_device=self.out_device, batch_sum=batch_sum, labels=labels)

    def auto_batchsize(self, func, ds_size=2**20, print_log=True):

        sample_one = self.sample_out(func)
        batchsize_est = min(auto_batchsize_output(self.HM.device, sample_one), ds_size)

        batchsize = int(batchsize_est)
        while True:
            if batchsize < 1:
                print('Memory consumption too large in application')
                exit()
            try:
                self.sample_out(func, batchsize)
                log('batchsize {} verified'.format(batchsize), print_log)
                break
            except:
                log('batchsize {} too large in application, reduce to half'.format(batchsize), print_log)
                batchsize = int(batchsize / 2)
        
        return batchsize

    def auto_dataloader(self, func, dscrop=1, print_log=True):

        batchsize = self.auto_batchsize(func, len(self.ds), print_log)
        self.dl.set_batchsize(batchsize)
        self.dl.set_dscrop(dscrop)
        return self.dl
    
    def write_cache(self, ind, func, res):

        name = "{}_{}".format(ind, func.__name__)
        self.cache[name] = res

    def load_cache(self, ind, func):
        
        name = "{}_{}".format(ind, func.__name__)
        if name in self.cache:
            if self.cache[name] is not None:
                if set(self.comp_layers) == set(self.cache[name].keys()):
                    log('Using cached {}'.format(name))
                    return self.cache[name]
        return None

class stats_(stats_base):

    def __init__(self, HM, comp_layers, out_device):
        super().__init__(HM, comp_layers, out_device)
        return
    
    def expectation(self, func, dscrop=1, out_device=None, from_cache=True, to_cache=True, print_log=True, **kwargs):

        if from_cache:
            cached_ret = self.load_cache('E', func)
            if cached_ret is not None:
                return cached_ret

        dataloader = self.auto_dataloader(func, dscrop, print_log)

        sample_count = 0
        if out_device is None:
            out_device
        st = datetime.datetime.now()
        log('Computing Expectation {} with batchsize {}'.format(func.__name__, dataloader.batchsize), print_log)
        
        batch_count = dataloader.batch_count
        report_inds = list(range(1, batch_count, max(1, int(batch_count/10))))
        layers = self.comp_layers
        ret = {}
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(self.HM.device)
            labels = labels.to(self.HM.device)
            sample_count += inputs.size()[0]
            sums = func(self.HM.ife, layers, inputs, self.HM.device, self.HM.Ws, out_device=out_device, batch_sum=True, labels=labels, **kwargs)
        
            for layer in layers:
                if layer not in ret:
                    ret[layer] = sums[layer]
                else:
                    ret[layer] += sums[layer]
            
            if print_log:
                if i in report_inds:
                    est(st, (i + 1) / batch_count)
        
        for layer in layers:
            ret[layer] /= sample_count
        if to_cache:
            self.write_cache('E', func, ret)
        
        timer(st, "\tDone Computing Expectation {} with {} samples".format(func.__name__, sample_count))
        return ret

    def variance(self, func, dscrop=1, out_device=None, from_cache=True, to_cache=True, print_log=True, **kwargs):

        if from_cache:
            cached_ret = self.load_cache('Var', func)
            if cached_ret is not None:
                return cached_ret
        
        E_vecs = self.expectation(func, dscrop, self.HM.device, from_cache, to_cache, print_log=print_log, **kwargs)
        square_vecs = self.expectation(func_square(func), dscrop, self.HM.device, from_cache, to_cache=False, print_log=print_log, **kwargs)
        
        ret = {}
        for layer in self.comp_layers:
            E_vec_square = E_vecs[layer].square()
            ret[layer] = square_vecs[layer].sub(E_vec_square)

        if to_cache:
            self.write_cache('Var', func, ret)
        return ret
    
    def covariance_matrix(self, func, dim, dscrop=1, out_device=None, from_cache=True, to_cache=True, print_log=True, **kwargs):

        """Compute the covariance matrix w.r.t. the column / row vectors
        For row vectors: dim=1, for column vectors: dim=0"""

        if from_cache:
            cached_ret = self.load_cache('Cov{}'.format(dim), func)
            if cached_ret is not None:
                return cached_ret

        layers = self.comp_layers
        sample_one = self.sample_out(func)
        for layer in layers:
            assert len(sample_one[layer].shape) == 3
        assert dim == 0 or dim == 1
        
        transpose_inds = (0, 1) if dim == 1 else (1, 0)
        auto_corr_func = chain_comp_((func, func), transpose_inds)

        E_vecs = self.expectation(func, dscrop, self.HM.device, from_cache, to_cache, print_log=print_log, **kwargs)
        auto_corr_mats = self.expectation(auto_corr_func, dscrop, self.HM.device, from_cache, to_cache=False, print_log=print_log, **kwargs)
        
        for layer in self.comp_layers:
            E_vec = E_vecs[layer]
            E_cross = torch.bmm(E_vec, E_vec.transpose(-1,-2)) if dim == 1 else torch.bmm(E_vec.transpose(-1,-2), E_vec) # pylint: disable=no-member
            auto_corr_mats[layer].sub_(E_cross)

        if to_cache:
            self.write_cache('Cov{}'.format(dim), func, auto_corr_mats)
        return auto_corr_mats

    

class TSA():
    
    class chain_comp(chain_comp_):
        def __init__(self, funcs, transposes=None):
            super().__init__(funcs, transposes)
    
    class stats(stats_):
        def __init__(self, HM, comp_layers, out_device):
            super().__init__(HM, comp_layers, out_device)
    
    class normalize_comp(normalize_comp_):
        def __init__(self, func, dim=-1):
            super().__init__(func, dim)


def cross_dot(func, Ms):

    def cross_dot_comp(ife, layers, inputs, device, Ws, out_device=None, batch_sum=False):
        func_rets = func(ife, layers, inputs, device, Ws, out_device=device)
        ret = {}
        for layer in layers:
            func_ret = func_rets[layer]
            M = Ms[layer].to(device).unsqueeze(0)
            ret[layer] = func_ret.matmul(M.transpose(-1,-2))
            if batch_sum:
                ret[layer] = ret[layer].sum(0)
            if out_device is not None:
                ret[layer] = ret[layer].to(out_device)
        empty_cache(device)
        return ret

    return cross_dot_comp

def cross_cos(func, Ms):

    def cross_cos_comp(ife, layers, inputs, device, Ws, out_device=None, batch_sum=False):
        func_rets = func(ife, layers, inputs, device, Ws, out_device=device)
        ret = {}
        for layer in layers:
            func_ret = func_rets[layer]
            func_ret = func_ret.div_(func_ret.norm(dim=-1).unsqueeze(-1))
            M = Ms[layer].to(device).unsqueeze(0)
            M = M.div_(M.norm(dim=-1).unsqueeze(-1))
            ret[layer] = func_ret.matmul(M.transpose(-1,-2))
            if batch_sum:
                ret[layer] = ret[layer].sum(0)
            if out_device is not None:
                ret[layer] = ret[layer].to(out_device)
        empty_cache(device)
        return ret

    return cross_cos_comp

def func_gram(func):
    def func_gram_comp(ife, layers, inputs, device, Ws, out_device=None, batch_sum=False):
        func_ret = func(ife, layers, inputs, device, Ws, out_device=device, batch_sum=False)
        ret = {}
        for layer in layers:
            func_vec = func_ret[layer]
            ret[layer] = func_vec.matmul(func_vec.transpose(-1, -2))
            if batch_sum:
                ret[layer] = ret[layer].sum(axis=0)
            if out_device is not None:
                ret[layer] = ret[layer].to(out_device)
        empty_cache(device)
        return ret
    
    return func_gram_comp

def func_square(func):
    def func_square_comp(ife, layers, inputs, device, Ws, out_device=None, batch_sum=False):
        func_ret = func(ife, layers, inputs, device, Ws, out_device=device, batch_sum=False)
        ret = {}
        for layer in layers:
            func_vec = func_ret[layer]
            ret[layer] = func_vec.square_()
            if batch_sum:
                ret[layer] = ret[layer].sum(axis=0)
            if out_device is not None:
                ret[layer] = ret[layer].to(out_device)
        empty_cache(device)
        return ret
    
    return func_square_comp