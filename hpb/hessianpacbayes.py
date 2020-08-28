import torch
import numpy as np
import math

from .dp import *
from .arithm import *
from .loss import PacBayesLoss, mnnLoss, PacBayesLoss_Hessian
from .test import eval_model
from .utils import *
import copy
from functools import reduce

class PBModule_base():

    def __init__(self, network: torch.nn.Module, datasets:list, criterion, accuracy_loss):

        """network and datasets [trainset, testset]"""

        self.net, self.device, self.device_handle = prepare_net(network) # type=torch.nn.Model, str, None
        self.criterion = criterion
        self.datasets = {True: datasets[0], False: datasets[1]}
        self.dataloaders = {train: OnDeviceDataLoader(self.datasets[train], 2048, device=self.device, shuffle=False) for train in [True, False]}
        self.accuracy_loss = accuracy_loss
        self.BRE = None
    
    def initialize_BRE_zero_prior(self,
                                  lambda_prior=-3.,
                                  conf_param=0.025,
                                  precision=100,
                                  bound=1
                                  ):

        mean_post = parameters_to_vector(self.net.parameters())
        mean_prior = torch.zeros(mean_post.shape) # pylint: disable=no-member

        sigma_post = torch.log(torch.Tensor.abs(mean_post)).to(self.device).requires_grad_() # pylint: disable=no-member
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable

        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.accuracy_loss, self.device).to(self.device)
    
    def initialize_BRE(self,
                       mean_prior,
                       lambda_prior=-3.,
                       conf_param=0.025,
                       precision=100,
                       bound=0.1
                       ):

        mean_post = parameters_to_vector(self.net.parameters())

        # sigma_post = torch.Tensor.abs(parameters_to_vector(self.net.parameters())).to(self.device).requires_grad_() # pylint: disable=no-member
        sigma_post = torch.log(torch.Tensor.abs(mean_post)).to(self.device).requires_grad_() # pylint: disable=no-member
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable

        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.accuracy_loss, self.device).to(self.device)

    def compute_bound(self,
                      n_monte_carlo_approx=1000,
                      delta_prime=0.01,
                      sample_freq=10):
        print("Using {} iteration Monte Carlo approx, delta'={}".format(n_monte_carlo_approx, delta_prime))
        snn_train_error, Pac_bound, kl = self.BRE.compute_bound(self.dataloaders[True], delta_prime, n_monte_carlo_approx, sample_freq)
        print(snn_train_error)
        print(Pac_bound)
        print('\nFinal PAC Bayes Bound: {:.4g};\nFinal SNN error: {:.4g};\nFinal KL Divergence: {:.5g}'.format(Pac_bound[-1], snn_train_error[-1], kl.item()))

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
        plog("state dict loaded from {}".format(1))

    def get_prior_mean(self, sd_path):
        net_tmp = copy.deepcopy(self.net)
        sd = torch.load(sd_path, map_location=self.device)
        net_tmp.load_state_dict(sd)
        return parameters_to_vector(net_tmp.parameters())

    def evaluate(self, train=True, log=False):
        ret = eval_model(self.net, self.dataloaders[train], self.criterion, self.accuracy_loss, self.device)
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
    PAC-Bayes with w as a naive estimation of post (mean w, std abs(w))
    """
    # TODO: Complete the first version of computation module. Done by Friday

    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)
        self.BRE = None
        return
    
class PacBayes_Optim(PBModule_base):
    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)
        return

    def optimize_PACB_RMSprop(self,
                              learning_rate=0.01,
                              alpha=0.9,
                              epoch_num=20,
                              lr_gamma=1,
                              lr_decay_mode='step',
                              batchsize=100,
                              step_lr_decay=100
                              ):
        
        """ Optimizing the PAC-Bayes Bound using RMSprop
        Parameters
        ----------
        learning_rate : float(default=0.01)
            Initial learning rate
        alpha : float (default=0.9)
            RMSprop coef
        epoch_num : int (default=100)
            number of epochs to optimize
        lr_gamma : int (default=0.1)
            learning rate decay rate for 10 epochs
        lr_decay_mode : str ['exp', 'step'] (default='exp')
            -
        batchsize : int
            -
        """

        assert self.BRE is not None, 'need to initialize BRE first'
        BRE = self.BRE

        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, BRE.parameters()), lr=learning_rate, alpha=0.9)
        nnloss = mnnLoss(self.net, self.criterion, BRE.mean_post, BRE.sigma_post_, BRE.d_size, self.device)

        train_loader = self.dataloaders[True]
        train_loader.set_batchsize(batchsize)
        
        if lr_decay_mode == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_lr_decay, gamma=lr_gamma)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma**(1 / 10), last_epoch=-1)

        t = time.time()
        BRE_losses, KL_value, SNN_losses, norm_weights, norm_sigma, norm_lambda = (list() for i in range(6))
        
        print("\nStarting PAC-Bayes bound optimization - RMSprop")
        train_info = "\nEpochs {}\ninitial_lr: {:.3g}\nlr_decay_alpha: {:.3g}\nlr_decay_mode: {}\nbatchsize: {}\n"
        print(train_info.format(epoch_num, learning_rate, lr_gamma, lr_decay_mode, batchsize))
        
        for epoch in np.arange(epoch_num):

            st = time.time()
            LOG_INFO = "Epoch {}: | BRE:{:.4g}, KL:{:.4g}, SNN_loss:{:.4g}, Std_prior:{:.4g}, lr:{:2g} | took {:4g}s"
            SNN_loss, BRE_loss = [], []
            for i, (inputs, labels) in enumerate(iter(train_loader)):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss1 = BRE()
                loss1.backward(retain_graph=True)

                noise = torch.randn(BRE.d_size).to(self.device) * torch.Tensor.exp(BRE.sigma_post_)
                loss2 = nnloss(inputs, labels, noise)
                BRE_loss.append(loss1.item())
                SNN_loss.append(loss2.item())

                self.net.zero_grad()
                loss2.backward()
                
                weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(nnloss.model.parameters())), dim=0) # pylint: disable=no-member
                BRE.mean_post.grad += weights_grad
                BRE.sigma_post_.grad += weights_grad * noise

                optimizer.step()
                optimizer.zero_grad()

            BRE_losses.append(np.mean(BRE_loss))
            SNN_losses.append(np.mean(SNN_loss))
            KL_value.append(BRE.kl_value)
            
            print(LOG_INFO.format(epoch, BRE_losses[-1], KL_value[-1], SNN_losses[-1], BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)))
            scheduler.step()
        
        plog("Optimization done. Took {:.4g}s".format(time.time() - t))

class PacBayes_Hessian(PBModule_base):
    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)

    def load_hessian(self, hessian_path):
        print(hessian_path)
        hessian_data = torch.load(hessian_path, map_location=self.device)
        self.eigenvals = hessian_data['eigenvals']
        self.eigenvecs = hessian_data['eigenvecs']
        self.eigenvecs_T = self.eigenvecs.transpose(0,1)
        plog("Hessian loaded from {}".format(1))
        
    def to_hessian(self, vec):
        return vec.matmul(self.eigenvecs_T)
    
    def to_standard(self, vec):
        return vec.matmul(self.eigenvecs)

    def initialize_BRE(self,
                       mean_prior,
                       lambda_prior=-3.,
                       conf_param=0.025,
                       precision=100,
                       bound=0.1
                       ):
        mean_post = parameters_to_vector(self.net.parameters())
        sigma_post = torch.log(torch.Tensor.abs(mean_post))
        sigma_post = sigma_post.to(self.device).requires_grad_() # pylint: disable=not-callable
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable

        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss_Hessian(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.accuracy_loss, self.device, self.to_standard).to(self.device)

    def initialize_BRE_eigenval(self,
                                mean_prior,
                                epsilon = 0.01,
                                lambda_prior=-3.,
                                conf_param=0.025,
                                precision=100,
                                bound=0.1
                                ):
        mean_post = parameters_to_vector(self.net.parameters())
        sigma_post = torch.log(torch.div(epsilon, torch.sqrt(self.eigenvals)))
        sigma_post[sigma_post > 0] = 0
        sigma_post[torch.isnan(sigma_post)] = 0
        print(list(sigma_post.to('cpu').numpy()))
        sigma_post = sigma_post.to(self.device).requires_grad_() # pylint: disable=not-callable
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable

        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss_Hessian(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.accuracy_loss, self.device, self.to_standard).to(self.device)
    
    def optimize_PACB_RMSprop(self,
                              learning_rate=0.01,
                              alpha=0.9,
                              epoch_num=20,
                              lr_gamma=1,
                              lr_decay_mode='step',
                              batchsize=100,
                              step_lr_decay=100
                              ):
        
        """ Optimizing the PAC-Bayes Bound using RMSprop
        Parameters
        ----------
        learning_rate : float(default=0.01)
            Initial learning rate
        alpha : float (default=0.9)
            RMSprop coef
        epoch_num : int (default=100)
            number of epochs to optimize
        lr_gamma : int (default=0.1)
            learning rate decay rate for 10 epochs
        lr_decay_mode : str ['exp', 'step'] (default='exp')
            -
        batchsize : int
            -
        """

        assert self.BRE is not None, 'need to initialize BRE first'
        BRE = self.BRE

        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, BRE.parameters()), lr=learning_rate, alpha=0.9)
        nnloss = mnnLoss(self.net, self.criterion, BRE.mean_post, BRE.sigma_post_, BRE.d_size, self.device)

        train_loader = self.dataloaders[True]
        train_loader.set_batchsize(batchsize)
        
        if lr_decay_mode == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_lr_decay, gamma=lr_gamma)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma**(1 / 10), last_epoch=-1)

        t = time.time()
        BRE_losses, KL_value, SNN_losses, norm_weights, norm_sigma, norm_lambda = (list() for i in range(6))
        
        print("\nStarting PAC-Bayes bound optimization - RMSprop")
        train_info = "\nEpochs {}\ninitial_lr: {:.3g}\nlr_decay_alpha: {:.3g}\nlr_decay_mode: {}\nbatchsize: {}\n"
        print(train_info.format(epoch_num, learning_rate, lr_gamma, lr_decay_mode, batchsize))
        
        for epoch in np.arange(epoch_num):

            st = time.time()
            LOG_INFO = "Epoch {}: | BRE:{:.4g}, KL:{:.4g}, SNN_loss:{:.4g}, Std_prior:{:.4g}, lr:{:2g} | took {:4g}s"
            SNN_loss, BRE_loss = [], []
            for i, (inputs, labels) in enumerate(iter(train_loader)):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss1 = BRE()
                loss1.backward(retain_graph=True)

                noise = torch.randn(BRE.d_size).to(self.device) * torch.Tensor.exp(BRE.sigma_post_)
                loss2 = nnloss(inputs, labels, self.to_standard(noise))
                BRE_loss.append(loss1.item())
                SNN_loss.append(loss2.item())

                self.net.zero_grad()
                loss2.backward()
                
                weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(nnloss.model.parameters())), dim=0) # pylint: disable=no-member
                BRE.mean_post.grad += weights_grad
                BRE.sigma_post_.grad += self.to_hessian(weights_grad) * noise

                optimizer.step()
                optimizer.zero_grad()

            BRE_losses.append(np.mean(BRE_loss))
            SNN_losses.append(np.mean(SNN_loss))
            KL_value.append(BRE.kl_value)
            
            print(LOG_INFO.format(epoch, BRE_losses[-1], KL_value[-1], SNN_losses[-1], BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)))
            scheduler.step()
        
        plog("Optimization done. Took {:.4g}s".format(time.time() - t))