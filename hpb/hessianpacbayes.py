import torch
import numpy as np
import math
from .dp import *
from .arithm import *
from .loss import *
from .test import eval_model
from .utils import *
from algos.closeformhessian import hessianmodule
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
        t = time.time()
        snn_train_error, Pac_bound, kl = self.BRE.compute_bound(self.dataloaders[True], delta_prime, n_monte_carlo_approx, sample_freq)
        print(snn_train_error)
        print(Pac_bound)
        print('\nFinal PAC Bayes Bound: {:.4g};\nFinal SNN error: {:.4g};\nFinal KL Divergence: {:.5g}'.format(Pac_bound[-1], snn_train_error[-1], kl.item()))
        plog("Compute Bound Done. Took {:.4g}s".format(time.time() - t))

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
        #BRE_losses, KL_value, SNN_losses, norm_weights, norm_sigma, norm_lambda = (list() for i in range(6))
        
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

            #BRE_losses.append(np.mean(BRE_loss))
            #SNN_losses.append(np.mean(SNN_loss))
            #KL_value.append(BRE.kl_value)
            
            #print(LOG_INFO.format(epoch, BRE_losses[-1], KL_value[-1], SNN_losses[-1], BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)), flush=True)
            print(LOG_INFO.format(epoch, np.mean(BRE_loss), BRE.kl_value, np.mean(SNN_loss), BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)), flush=True)
            scheduler.step()
        
        plog("Optimization done. Took {:.4g}s".format(time.time() - t))

class PacBayes_Hessian(PBModule_base):
    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)

    def load_hessian_file(self, hessian_path):
        print(hessian_path)
        hessian_data = torch.load(hessian_path, map_location=self.device)
        self.eigenvals = hessian_data['eigenvals']
        self.eigenvecs = hessian_data['eigenvecs']
        self.eigenvecs_T = self.eigenvecs.transpose(0,1)
        plog("Hessian loaded from {}".format(1))
    
    def load_eigenthings(self, eigenvals, eigenvecs):
        self.eigenvals = eigenvals.to(self.device)
        self.eigenvecs = eigenvecs.to(self.device)
        self.eigenvecs_T = self.eigenvecs.transpose(0,1)
        
    def to_hessian(self, vec):
        return vec.matmul(self.eigenvecs_T)
    
    def to_standard(self, vec):
        return vec.matmul(self.eigenvecs)

    def noise_generation(self):
        noise_hessian = torch.randn(self.BRE.d_size).to(self.device) * torch.Tensor.exp(self.BRE.sigma_post_)
        noise_standard = self.to_standard(noise_hessian)
        return noise_standard, noise_hessian

    def initialize_BRE(self,
                       mean_prior,
                       lambda_prior=-3.,
                       conf_param=0.025,
                       precision=100,
                       bound=0.1
                       ):
        mean_post = parameters_to_vector(self.net.parameters())
        self.original_mean_post = copy.deepcopy(mean_post.detach())
        print(mean_post)
        sigma_post = torch.log(torch.Tensor.abs(self.to_hessian(mean_post)))
        #print(list(sigma_post.detach().to('cpu').numpy()))
        sigma_post = sigma_post.to(self.device).requires_grad_() # pylint: disable=not-callable
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable
        
        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss_Hessian(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.accuracy_loss, self.device, self.noise_generation).to(self.device)

    def initialize_BRE_eigenval(self,
                                mean_prior,
                                epsilon = None,
                                lambda_prior=-3.,
                                conf_param=0.025,
                                precision=100,
                                bound=0.1
                                ):
        if epsilon is None:
            epsilon = np.exp(2*lambda_prior)
        mean_post = parameters_to_vector(self.net.parameters())
        sigma_post = torch.log(torch.div(epsilon, self.eigenvals.abs().sqrt()))
        sigma_post[sigma_post > 0] = 0
        sigma_post[torch.isnan(sigma_post)] = 0
        #print(list(sigma_post.to('cpu').numpy()))
        sigma_post = sigma_post.to(self.device).requires_grad_() # pylint: disable=not-callable
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable

        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss_Hessian(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.accuracy_loss, self.device, self.noise_generation).to(self.device)
    
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
        #BRE_losses, KL_value, SNN_losses, norm_weights, norm_sigma, norm_lambda = (list() for i in range(6))
        
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

                noise_standard, noise_hessian = self.noise_generation()
                loss2 = nnloss(inputs, labels, noise_standard)
                BRE_loss.append(loss1.item())
                SNN_loss.append(loss2.item())

                self.net.zero_grad()
                loss2.backward()
                
                weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(nnloss.model.parameters())), dim=0) # pylint: disable=no-member
                BRE.mean_post.grad += weights_grad
                BRE.sigma_post_.grad += self.to_hessian(weights_grad) * noise_hessian

                optimizer.step()
                optimizer.zero_grad()

            #BRE_losses.append(np.mean(BRE_loss))
            #SNN_losses.append(np.mean(SNN_loss))
            #KL_value.append(BRE.kl_value)
            
            #print(LOG_INFO.format(epoch, BRE_losses[-1], KL_value[-1], SNN_losses[-1], BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)), flush=True)
            print(LOG_INFO.format(epoch, np.mean(BRE_loss), BRE.kl_value, np.mean(SNN_loss), BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)), flush=True)
            scheduler.step()
        
        plog("Optimization done. Took {:.4g}s".format(time.time() - t))

class PacBayes_Hessian_partial(PacBayes_Hessian):
    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)

    def to_hessian(self, vec):
        ans = vec.matmul(self.eigenvecs_T)
        res = vec.sub(ans.matmul(self.eigenvecs))
        val = res.norm().div_(math.sqrt(vec.shape[0]-ans.shape[0]))
        return torch.cat((ans, val.unsqueeze(0)))
    
    def noise_generation(self):
        rand_vec = torch.randn(self.BRE.d_size).to(self.device)
        proj_vec = rand_vec.matmul(self.eigenvecs_T)
        res_vec = rand_vec.sub(proj_vec.matmul(self.eigenvecs))
        res_val = res_vec.norm().div_(math.sqrt(rand_vec.shape[0]-proj_vec.shape[0]))
        noise_hessian = torch.cat((proj_vec, res_val.unsqueeze(0)))
        #print(list(self.BRE.sigma_post_.detach().to('cpu').numpy()))
        noise_hessian = noise_hessian.mul_(torch.exp(self.BRE.sigma_post_))
        noise_standard = noise_hessian[:-1].matmul(self.eigenvecs)
        noise_standard = noise_standard.add_(res_vec.div(res_val).mul(noise_hessian[-1]))
        return noise_standard, noise_hessian

    def initialize_BRE_eigenval(self,
                                mean_prior,
                                epsilon = None,
                                lambda_prior=-3.,
                                conf_param=0.025,
                                precision=100,
                                bound=0.1
                                ):
        if epsilon is None:
            epsilon = np.exp(2*lambda_prior)
        mean_post = parameters_to_vector(self.net.parameters())
        sigma_post = torch.log(torch.div(epsilon, self.eigenvals.abs().sqrt()))
        sigma_post[sigma_post > 0] = 0
        sigma_post[torch.isnan(sigma_post)] = 0
        sigma_post = torch.cat((sigma_post, torch.Tensor([0]).to(self.device)))
        #print(list(sigma_post.to('cpu').numpy()))
        sigma_post = sigma_post.to(self.device).requires_grad_() # pylint: disable=not-callable
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable

        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss_Hessian_partial(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.accuracy_loss, self.device, self.noise_generation).to(self.device)
    
class PacBayes_Hessian_layerwise(PacBayes_Hessian):
    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)

    def load_hessian_file(self, hessian_path):
        print(hessian_path)
        hessian_data = torch.load(hessian_path, map_location=self.device)
        self.layers = hessian_data['layers']
        self.H_eigenvals = hessian_data['H_eigenvals']
        self.H_eigenvecs = hessian_data['H_eigenvecs']
        self.UTAU_eigenvals = hessian_data['UTAU_eigenvals']
        self.UTAU_eigenvecs = hessian_data['UTAU_eigenvecs']
        self.eigenvals = []
        self.H_eigenvecs_T = {}
        self.UTAU_eigenvecs_T = {}
        self.H_d = {}
        self.UTAU_d = {}
        for layer in self.layers:
            self.eigenvals.append(self.H_eigenvals[layer])
            self.eigenvals.append(self.UTAU_eigenvals[layer])
            self.H_eigenvecs_T[layer] = self.H_eigenvecs[layer].t()
            self.UTAU_eigenvecs_T[layer] = self.UTAU_eigenvecs[layer].t()
            self.H_d[layer] = self.H_eigenvecs[layer].shape[1]
            self.UTAU_d[layer] = self.UTAU_eigenvecs[layer].shape[1]
        self.eigenvals = torch.cat(self.eigenvals)
        plog("Hessian loaded from {}".format(1))

    def to_hessian(self, vec):
        ans = []
        s = 0
        for layer in self.layers:
            e = s + self.H_d[layer]
            ans.append(vec[s:e].matmul(self.H_eigenvecs_T[layer]))
            s = e
            e = s + self.UTAU_d[layer]
            ans.append(vec[s:e].matmul(self.UTAU_eigenvecs_T[layer]))
            s = e
        return torch.cat(ans)
    
    def to_standard(self, vec):
        ans = []
        s = 0
        for layer in self.layers:
            e = s + self.H_d[layer]
            ans.append(vec[s:e].matmul(self.H_eigenvecs[layer]))
            s = e
            e = s + self.UTAU_d[layer]
            ans.append(vec[s:e].matmul(self.UTAU_eigenvecs[layer]))
            s = e
        return torch.cat(ans)

class PacBayes_Hessian_approx(PacBayes_Hessian):
    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)

    def load_hessian_file(self, hessian_path):
        print(hessian_path)
        hessian_data = torch.load(hessian_path, map_location=self.device)
        self.layers = hessian_data['layers']
        self.UTAU_eigenvals = hessian_data['UTAU_eigenvals']
        self.UTAU_eigenvecs = hessian_data['UTAU_eigenvecs']
        self.xxT_eigenvals = hessian_data['xxT_eigenvals']
        self.xxT_eigenvecs = hessian_data['xxT_eigenvecs']
        self.norms = hessian_data['norms']
        self.eigenvals = []
        self.UTAU_eigenvecs_T = {}
        self.xxT_eigenvecs_T = {}
        self.UTAU_d = {}
        self.xxT_d = {}
        for layer in self.layers:
            self.eigenvals.append(self.UTAU_eigenvals[layer].ger(self.xxT_eigenvals[layer]).reshape(-1))
            self.eigenvals.append(self.UTAU_eigenvals[layer])
            self.UTAU_eigenvecs_T[layer] = self.UTAU_eigenvecs[layer].t()
            self.xxT_eigenvecs_T[layer] = self.xxT_eigenvecs[layer].t()
            self.UTAU_d[layer] = self.UTAU_eigenvecs[layer].shape[1]
            self.xxT_d[layer] = self.xxT_eigenvecs[layer].shape[1]
        self.eigenvals = torch.cat(self.eigenvals)
        plog("Hessian loaded from {}".format(1))
    
    def hessian_calc(self, network, layers, y_classification_mode='binary_logistic_pn1'):
        self.layers = layers
        self.y_classification_mode = y_classification_mode
        HM = hessianmodule.HessianModule(network, self.datasets[True], self.layers, RAM_cap=64, print_log=False)
        self.HM = HM
        HM.load_sd(self.get_sd())
        UTAU = HM.expectation(HM.decomp.UTAU_comp, self.layers, out_device=HM.device, to_cache=False, print_log=False, y_classification_mode=y_classification_mode)
        xxT = HM.expectation(HM.decomp.xxT_comp, self.layers, out_device=HM.device, to_cache=False, print_log=False)
        self.UTAU_eigenvals, self.UTAU_eigenvecs, self.UTAU_eigenvecs_T, self.UTAU_d = {}, {}, {}, {}
        self.xxT_eigenvals, self.xxT_eigenvecs, self.xxT_eigenvecs_T, self.xxT_d = {}, {}, {}, {}
        self.norms = {}
        self.eigenvals = []
        for layer in self.layers:
            self.UTAU_eigenvals[layer], self.UTAU_eigenvecs[layer] = HM.utils.eigenthings_tensor_utils(UTAU[layer], device=HM.device, symmetric=True)
            self.xxT_eigenvals[layer], self.xxT_eigenvecs[layer] = HM.utils.eigenthings_tensor_utils(xxT[layer], device=HM.device, symmetric=True) 
            self.eigenvals.append(self.UTAU_eigenvals[layer].ger(self.xxT_eigenvals[layer]).reshape(-1))
            self.eigenvals.append(self.UTAU_eigenvals[layer])
            self.UTAU_eigenvecs_T[layer] = self.UTAU_eigenvecs[layer].t()
            self.xxT_eigenvecs_T[layer] = self.xxT_eigenvecs[layer].t()
            self.UTAU_d[layer] = self.UTAU_eigenvecs[layer].shape[1]
            self.xxT_d[layer] = self.xxT_eigenvecs[layer].shape[1]
            ones_mat = torch.ones((self.UTAU_d[layer], self.xxT_d[layer])).to(HM.device)
            self.norms[layer] = self.UTAU_eigenvecs[layer].square().matmul(ones_mat).matmul(self.xxT_eigenvecs_T[layer].square())
            self.norms[layer] = self.norms[layer].reshape(-1).sqrt_()
        self.eigenvals = torch.cat(self.eigenvals)
        print("Hessian Calculation Complete", flush=True)
        self.old_UTAU = UTAU
        self.old_xxT = xxT
        self.old_UTAU_eigenvecs = copy.deepcopy(self.UTAU_eigenvecs)
        self.old_xxT_eigenvecs = copy.deepcopy(self.xxT_eigenvecs)
    def to_hessian(self, vec):
        ans = []
        s = 0
        for layer in self.layers:
            e = s + self.UTAU_d[layer] * self.xxT_d[layer]
            vec_mat = vec[s:e].reshape(self.UTAU_d[layer], self.xxT_d[layer])
            ans_mat = self.UTAU_eigenvecs[layer].matmul(vec_mat).matmul(self.xxT_eigenvecs_T[layer])
            ans.append(ans_mat.reshape(-1).div_(self.norms[layer]))
            s = e
            e = s + self.UTAU_d[layer]
            ans.append(vec[s:e].matmul(self.UTAU_eigenvecs_T[layer]))
            s = e
        return torch.cat(ans)
    
    def to_standard(self, vec):
        ans = []
        s = 0
        for layer in self.layers:
            e = s + self.UTAU_d[layer] * self.xxT_d[layer]
            vec_mat = vec[s:e].mul(self.norms[layer]).reshape(self.UTAU_d[layer], self.xxT_d[layer])
            ans_mat = self.UTAU_eigenvecs_T[layer].matmul(vec_mat).matmul(self.xxT_eigenvecs[layer])
            ans.append(ans_mat.reshape(-1))
            s = e
            e = s + self.UTAU_d[layer]
            ans.append(vec[s:e].matmul(self.UTAU_eigenvecs[layer]))
            s = e
        return torch.cat(ans)

class PacBayes_Hessian_iterative(PacBayes_Hessian_approx):
    def __init__(self, network, datasets, criterion, accuracy_loss):
        super().__init__(network, datasets, criterion, accuracy_loss)

    def iterative_hessian_calc(self):
        HM = self.HM
        mean_post = self.BRE.mean_post.detach()
        mean_diff = torch.norm(self.original_mean_post-mean_post)
        hessian_diff = 0

        #print(self.BRE.sigma_post_.detach())
        sigma_post_standard = self.to_standard(self.BRE.sigma_post_.detach())
        #print(self.to_hessian(sigma_post_standard))
        #HM.load_vec(mean_post)
        #HM.load_sd(self.get_sd())
        UTAU = HM.expectation(HM.decomp.UTAU_comp, self.layers, out_device=HM.device, to_cache=False, print_log=False, y_classification_mode=self.y_classification_mode)
        xxT = HM.expectation(HM.decomp.xxT_comp, self.layers, out_device=HM.device, to_cache=False, print_log=False)
        for layer in self.layers: 
            self.UTAU_eigenvals[layer], self.UTAU_eigenvecs[layer] = HM.utils.eigenthings_tensor_utils(UTAU[layer], device=HM.device, symmetric=True)
            self.xxT_eigenvals[layer], self.xxT_eigenvecs[layer] = HM.utils.eigenthings_tensor_utils(xxT[layer], device=HM.device, symmetric=True) 
            self.UTAU_eigenvecs_T[layer] = self.UTAU_eigenvecs[layer].t()
            self.xxT_eigenvecs_T[layer] = self.xxT_eigenvecs[layer].t()
            self.UTAU_d[layer] = self.UTAU_eigenvecs[layer].shape[1]
            self.xxT_d[layer] = self.xxT_eigenvecs[layer].shape[1]
            ones_mat = torch.ones((self.UTAU_d[layer], self.xxT_d[layer])).to(HM.device)
            self.norms[layer] = self.UTAU_eigenvecs[layer].square().matmul(ones_mat).matmul(self.xxT_eigenvecs_T[layer].square())
            self.norms[layer] = self.norms[layer].reshape(-1).sqrt_()

            #hessian_diff += torch.norm(self.UTAU_eigenvecs[layer]-self.old_UTAU_eigenvecs[layer])
            #hessian_diff += torch.norm(self.xxT_eigenvecs[layer]-self.old_xxT_eigenvecs[layer])
            #print(torch.norm(self.UTAU_eigenvecs[layer]-self.old_UTAU_eigenvecs[layer]))
            #print(torch.norm(self.xxT_eigenvecs[layer]-self.old_xxT_eigenvecs[layer]))
            #print(torch.norm(UTAU[layer]-self.old_UTAU[layer]))
            #print(torch.norm(xxT[layer]-self.old_xxT[layer]))
        self.BRE.sigma_post_ = nn.Parameter(self.to_hessian(sigma_post_standard))
        #print(self.to_hessian(sigma_post_standard))
        #print(self.BRE.sigma_post_.detach())
        #print('mean difference: {:.4g}, hessian difference: {:.4g}'.format(mean_diff, hessian_diff), flush=True)
    
    def optimize_PACB_RMSprop(self,
                              learning_rate=0.01,
                              alpha=0.9,
                              epoch_num=20,
                              lr_gamma=1,
                              lr_decay_mode='step',
                              batchsize=100,
                              step_lr_decay=100,
                              hessian_calc_interval=1,
                              hessian_calc_decay=10
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
        #BRE_losses, KL_value, SNN_losses, norm_weights, norm_sigma, norm_lambda = (list() for i in range(6))
        
        print("\nStarting PAC-Bayes bound optimization - RMSprop")
        train_info = "\nEpochs {}\ninitial_lr: {:.3g}\nlr_decay_alpha: {:.3g}\nlr_decay_mode: {}\nbatchsize: {}\n"
        print(train_info.format(epoch_num, learning_rate, lr_gamma, lr_decay_mode, batchsize))
        
        hessian_calc_epoch = 1
        for epoch in np.arange(epoch_num):
            st = time.time()

            if epoch % step_lr_decay == 0 and epoch > 0:
                hessian_calc_interval *= hessian_calc_decay
            if epoch == hessian_calc_epoch:
                self.iterative_hessian_calc()
                hessian_calc_epoch += hessian_calc_interval

            
            LOG_INFO = "Epoch {}: | BRE:{:.4g}, KL:{:.4g}, SNN_loss:{:.4g}, Std_prior:{:.4g}, lr:{:2g} | took {:4g}s"
            SNN_loss, BRE_loss = [], []
            for i, (inputs, labels) in enumerate(iter(train_loader)):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss1 = BRE()
                loss1.backward(retain_graph=True)

                noise_standard, noise_hessian = self.noise_generation()
                loss2 = nnloss(inputs, labels, noise_standard)
                BRE_loss.append(loss1.item())
                SNN_loss.append(loss2.item())

                self.net.zero_grad()
                loss2.backward()
                
                weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(nnloss.model.parameters())), dim=0) # pylint: disable=no-member
                BRE.mean_post.grad += weights_grad
                BRE.sigma_post_.grad += self.to_hessian(weights_grad) * noise_hessian

                optimizer.step()
                optimizer.zero_grad()

            #BRE_losses.append(np.mean(BRE_loss))
            #SNN_losses.append(np.mean(SNN_loss))
            #KL_value.append(BRE.kl_value)
            
            #print(LOG_INFO.format(epoch, BRE_losses[-1], KL_value[-1], SNN_losses[-1], BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)), flush=True)
            print(LOG_INFO.format(epoch, np.mean(BRE_loss), BRE.kl_value, np.mean(SNN_loss), BRE.lambda_prior_, scheduler.get_last_lr()[0], float(time.time() - st)), flush=True)
            scheduler.step()
        
        plog("Optimization done. Took {:.4g}s".format(time.time() - t))