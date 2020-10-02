import os, sys
import torch
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .utils import *
from .dp import *
from .decomposition import *
from .measures import Measures
from .decomposition import Decomp

from .tsa import *
from .visualization import vis
from .layerwisefeature import IntermediateFeatureExtractor

class HessianModule():

    def __init__(self, net, dataset, fc_seq, use_gpu=True, RAM_cap=64, net_eval=True, net_prepare=True, device=None, remain_labels=None, on_device_dataloader=True):
        
        self.dataset = dataset
        if not net_prepare:
            assert device is not None
            self.device = device
        else:
            net, self.device, _ = prepare_net(net, use_gpu)
        if net_eval:
            net.eval()

        self.fc_seq = fc_seq
        self.sd = net.state_dict()
        self.ife = IntermediateFeatureExtractor(net, self.fc_seq)

        self.Ws = None
        self.RAM_cap = RAM_cap * (2 ** 30)
        self.load_Ws()

        self.measure = Measures(self.device)
        self.vis = vis(self.device)
        self.utils = Utils(self, self.device)
        self.decomp = Decomp()
        self.tsa = TSA()

        self.cache = {}
        self.remain_labels = remain_labels
        self.dl = OnDeviceDataLoader(self.dataset, 2048, self.device)

        self.tsa_sm = self.tsa.stats(self, fc_seq, 'cpu')
        gpu_memory()
        print("Layers to be evaluated: {}".format(fc_seq))
    
    def load_Ws(self):
        sample_data = sample_input(self.dataset)[0].to(self.device)
        _, sample_out = self.ife(sample_data)
        self.Ws = weight_load(self.ife.net.state_dict(), self.fc_seq, sample_out)
        del sample_data, sample_out
        empty_cache(self.device)

    def load_sd(self, sd):
        log("Loaded state dict")
        self.ife.net.load_state_dict(sd)
        self.clear_cache()
        self.load_Ws()

    def get_vec(self):
        return parameters_to_vector(self.ife.net.parameters())

    def load_vec(self, vec):
        log("Loaded parameter vector")
        vector_to_parameters(vec, self.ife.net.parameters())
        self.clear_cache()
        self.load_Ws()
 
    def set_remain_labels(self, remain_labels):
        self.dl.set_remain_labels(remain_labels)
    
    def clear_cache(self):
        log('cache cleared')
        self.cache = {}
    
    def config_stats_module(self, comp_layers, out_device):
        self.tsa_sm.comp_layers = comp_layers
        self.tsa_sm.out_device = out_device

    def comp_output(self, func, comp_layers, inputs, out_device='cpu', batch_sum=False, **kwargs):
        inputs = inputs.to(self.device)
        return func(self.ife, comp_layers, inputs, self.device, self.Ws, out_device=out_device, batch_sum=batch_sum, **kwargs)

    def sample_output(self, func, comp_layers, sample_count=1, out_device='cpu', batch_sum=False, **kwargs):
        inputs, labels = sample_input(self.dataset, sample_count, remain_labels=self.remain_labels)
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        print(labels)
        return func(self.ife, comp_layers, inputs, self.device, self.Ws, out_device=out_device, batch_sum=batch_sum, labels=labels, **kwargs)
    
    def expectation(self, func, comp_layers, out_device='cpu', dscrop=1, from_cache=True, to_cache=True, print_log=True, batchsize=None, **kwargs):
        self.config_stats_module(comp_layers, out_device)
        return self.tsa_sm.expectation(func, dscrop=dscrop, from_cache=from_cache, to_cache=to_cache, print_log=print_log, batchsize=batchsize, **kwargs)
    
    def variance(self, func, comp_layers, out_device='cpu', dscrop=1, from_cache=True, to_cache=True, print_log=True, **kwargs):
        self.config_stats_module(comp_layers, out_device)
        return self.tsa_sm.variance(func, dscrop=dscrop, from_cache=from_cache, to_cache=to_cache, print_log=print_log, **kwargs)

    def covariance(self, func, comp_layers, dim, out_device='cpu', dscrop=1, from_cache=True, to_cache=True, print_log=True, **kwargs):
        """row vectors: dim=1, col vectors: dim=0"""
        self.config_stats_module(comp_layers, out_device)
        return self.tsa_sm.covariance_matrix(func, dim, dscrop=dscrop, from_cache=from_cache, to_cache=to_cache, print_log=print_log, **kwargs)

    def hessian_eigenthings_estimate(self, comp_layers, out_device='cpu', dataset_crop=1, num_eigenthings=100, seed=None, **kwargs):
        torch.manual_seed(get_time_seed() if seed is None else seed)
        E_UTAUs = self.expectation(self.decomp.UTAU_comp, comp_layers, out_device, dataset_crop, **kwargs)
        E_xxTs = self.expectation(self.decomp.xxT_comp, comp_layers, out_device, dataset_crop, **kwargs)
        ret = {layer: self.decomp.eigenthings_exp_hessian_approx(E_UTAUs[layer], E_xxTs[layer], num_eigenthings, self.device, out_device) for layer in comp_layers}
        return ret