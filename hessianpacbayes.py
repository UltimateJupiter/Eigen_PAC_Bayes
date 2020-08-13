import torch
from .utils import *

class PBModule_base_():

    def __init__(self, network, dataset, use_gpu=True):

        self.net, self.device, self.device_handle = prepare_net(network) # type=torch.nn.Model, str, None
        self.ds = dataset
    
    def get_sd(self, out_device=None):
        if out_device is None:
            out_device = self.device
        sd = self.net.state_dict()
        for k in sd.keys():
            sd[k] = sd[k].to(out_device)
        return sd
        

    def train(self):
        return
        # TODO: Optional, implement to train the model from random initialization

    def generalization_gap(self, model, sd):
        return
        # TODO: given self.ds, compute the generalization gap

class PacBayes_Normal(PBModule_base_):
    """
    PAC-Bayes with w as a naive estimation of posterior (mean w, std abs(w))
    """
    # TODO: Complete the first version of computation module. Done by Friday

    def __init__(self, network, dataset, use_gpu):
        super().__init__(network, dataset, use_gpu)
        return
    
    