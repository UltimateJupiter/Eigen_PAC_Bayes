import torch
from .dp import *
from .arithm import *
from .loss import PacBayesLoss, mnnLoss
from .test import eval_model
from .utils import *
from functools import reduce

class PBModule_base():

    def __init__(self, network: torch.nn.Module, datasets:list):

        """network and datasets [trainset, testset]"""

        self.net, self.device, self.device_handle = prepare_net(network) # type=torch.nn.Model, str, None
        self.criterion = torch.nn.CrossEntropyLoss()
        self.datasets = {True: datasets[0], False: datasets[1]}
        self.dataloaders = {train: OnDeviceDataLoader(self.datasets[train], 2048, device=self.device, shuffle=False) for train in [True, False]}
        self.BRE = None
    
    def initialize_BRE(self,
                       lambda_prior=-3.,
                       conf_param=0.025,
                       precision=100,
                       bound=1
                       ):

        mean_post = parameters_to_vector(self.net.parameters())
        mean_prior = torch.zeros(mean_post.shape) # pylint: disable=no-member

        sigma_post = torch.Tensor.abs(parameters_to_vector(self.net.parameters())).to(self.device).requires_grad_() # pylint: disable=no-member
        lambda_prior = torch.tensor(lambda_prior, device=self.device).requires_grad_() # pylint: disable=not-callable

        data_size = len(self.datasets[True])

        self.BRE = PacBayesLoss(self.net, mean_prior, lambda_prior, mean_post, sigma_post, conf_param, precision, bound, data_size, self.device).to(self.device)
    
    def compute_bound(self,
                    n_monte_carlo_approx=100,
                    delta_prime=0.01):
        snn_train_error, Pac_bound, kl = self.BRE.compute_bound(self.dataloaders[True], delta_prime, n_monte_carlo_approx)
        print(snn_train_error, Pac_bound, kl)

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
    PAC-Bayes with w as a naive estimation of post (mean w, std abs(w))
    """
    # TODO: Complete the first version of computation module. Done by Friday

    def __init__(self, network, datasets):
        super().__init__(network, datasets)
        self.BRE = None
        return
    
    

class PacBayes_Optim(PacBayes_Naive):
    def __init__(self, network, datasets):
        super().__init__(network, datasets)
        return

    def optimize_PACB_RMSprop(self,
                              learning_rate=0.01,
                              alpha=0.9,
                              epoch_num=100,
                              lr_gamma=0.1,
                              lr_decay_mode='step',
                              batchsize=100
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

        self.initialize_BRE()
        BRE = self.BRE

        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, BRE.parameters()), lr=learning_rate, alpha=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        
        nnloss = mnnLoss(self.net, torch.nn.CrossEntropyLoss(), BRE.mean_post, BRE.sigma_post_, BRE.d_size, self.device)
        train_loader = self.dataloaders[True]
        train_loader.set_batchsize(batchsize)
        
        if lr_decay_mode == 'exp':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=lr_gamma)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma**(1 / 10), last_epoch=-1)

        t = time.time()
        mean_losses, BRE_loss, KL_value, NN_loss_final, norm_weights, norm_sigma, norm_lambda, outputs = (list() for i in range(8))
        
        print("\nStarting PAC-Bayes bound optimization - RMSprop")
        
        for epoch in np.arange(epoch_num):

            LOG_INFO = "Epoch {}: BRE:{:.4g}, KL:{:.4g}, SNN_loss:{:.4g}, lr:{:2g}"
            NN_loss = list()
                
            for i, (inputs, labels) in enumerate(iter(train_loader)):
                if i == ((BRE.data_size * 5) / 100):
                    print("\r Progress: {}%".format(100 * i // BRE.data_size), end="")

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss1 = BRE()
                loss1.backward(retain_graph=True)

                loss2 = nnloss(inputs, labels)
                loss = loss1 + loss2
                NN_loss.append(loss2)

                if (i == ((BRE.data_size * 5) / 100) and ((100 * i // BRE.data_size) - (100 * 5 * (i-1) // BRE.data_size)) != 0 and i != 0): 
                    print('\t Mean loss : {} \r'.format(sum(mean_losses) / len(mean_losses)))
                    mean_losses = []
                else:
                    mean_losses.append(loss.item())
                    
                self.net.zero_grad()
                loss2.backward()

                # Optimization Step
                weights_grad = torch.cat(list(Z.grad.view(-1) for Z in list(nnloss.model.parameters())), dim=0) # pylint: disable=no-member
                BRE.mean_post.grad += weights_grad
                BRE.sigma_post_.grad += weights_grad * nnloss.noise 

                optimizer.step()
                optimizer.zero_grad()

            BRE_loss.append(loss1.item())
            KL_value.append(BRE.kl_value)
            NN_loss_final.append(reduce(lambda a, b : a + b, NN_loss) / len(NN_loss))
            norm_weights.append(torch.norm(BRE.mean_post.clone().detach(), p=2))
            norm_sigma.append(torch.norm(BRE.sigma_post_.clone().detach(), p=2))
            norm_lambda.append(torch.Tensor.abs(BRE.lambda_prior_.clone().detach()))
            # plog(LOG_INFO.format(epoch, BRE_loss[-1], KL_value[-1], NN_loss_final[-1], scheduler.get_last_lr()[0]))
            plog((epoch, BRE_loss[-1], KL_value[-1], NN_loss_final[-1], scheduler.get_last_lr()[0]))
            scheduler.step()
        
        plog("Optimization done")
        print("Computation time is {}".format(time.time() - t))
        print("\n==> Saving Parameters... ")