import torch
from .utils import *
from .dp import *
from .arithm import *
from .loss import PacBayesLoss, mnnLoss
from .test import eval_model

class PBModule_base():

    def __init__(self, network: torch.nn.Module, datasets:list):

        """network and datasets [trainset, testset]"""

        self.net, self.device, self.device_handle = prepare_net(network) # type=torch.nn.Model, str, None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.datasets = {True: datasets[0], False: datasets[1]}
        self.dataloaders = {train: OnDeviceDataLoader(self.datasets[train], 2048, device=self.device, shuffle=False) for train in [True, False]}
    
    def get_sd(self, out_device=None):
        if out_device is None:
            out_device = self.device
        sd = self.net.state_dict()
        for k in sd.keys():
            sd[k] = sd[k].to(out_device)
        return sd
    
    def load_sd(self, sd_path):
        print(sd_path)
        sd = torch.load(sd_path, map_location=self.device)
        self.net.load_state_dict(sd)
        log("state dict loaded from {}".format(sd_path))

    def evaluate(self, train=True, log=False):
        ret = eval_model(self.net, self.dataloaders[train], self.criterion, self.device)
        if log:
            print(ret)
        return ret
    
    def generalization_gap(self):
        train_res = self.evaluate(True)
        test_res = self.evaluate(False)
        gap = dict()
        for k in train_res.keys():
            gap[k] = train_res[k] - test_res[k]
        return gap

    def train(self):
        return
        # TODO: Optional, implement to train the model from random initialization

class PacBayes_Naive(PBModule_base):
    """
    PAC-Bayes with w as a naive estimation of posterior (mean w, std abs(w))
    """
    # TODO: Complete the first version of computation module. Done by Friday

    def __init__(self, network, datasets):
        super().__init__(network, datasets)
        return
    
    def naive_bound(self,
                    lambda_prior=-3,
                    conf_param=0.025,
                    precision=100,
                    bound=1,
                    n_monte_carlo_approx=100,
                    delta_prime=0.01):

        lambda_prior = torch.Tensor(-3., device=self.device).requires_grad_() # pylint: disable=no-member
        sigma_posterior = torch.abs(parameters_to_vector(self.net.parameters())).to(self.device).requires_grad_() # pylint: disable=no-member
        w_vec = parameters_to_vector(self.net.parameters())
        initial_weights = torch.zeros(w_vec.shape) # pylint: disable=no-member

        data_size = len(self.datasets[True])

        BRE = PacBayesLoss(lambda_prior, sigma_posterior, self.net, w_vec, conf_param, precision, bound, data_size, initial_weights, self.device).to(self.device)
        snn_train_error, Pac_bound, kl = BRE.compute_bound(self.dataloaders[True], delta_prime, n_monte_carlo_approx)
        print(snn_train_error, Pac_bound, kl)