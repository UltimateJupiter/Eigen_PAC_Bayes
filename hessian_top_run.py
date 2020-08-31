import sys, os
import torch
from hpb.hessianpacbayes import *

sd_sgd_sol = 'experiment_log/run_1/models/final.pth'
sd_init = 'experiment_log/run_1/models/epoch0.pth'
hessian_file = 'experiment_log/run_1/models/final.pth_ET1000.eval'

test_dirc = '../hessian_eigenspace_overlap/MNIST_Binary/experiments/FC1_600_sgd0.01m0.9LS_l1d_pic01_labelpn1_bt100'
top_n = 1000

def main():

    assert os.path.isdir(test_dirc)
    sys.path.append(test_dirc)
    from config import Config # pylint: disable=no-name-in-module
    conf = Config()

    net = conf.net()
    criterion_nn = conf.criterion
    accuracy_loss = conf.accuracy_loss
    datasets = [conf.dataset(train=True, transform=conf.train_transform), conf.dataset(train=False, transform=conf.train_transform)]
    sd_path_sgd_sol, sd_path_init = os.path.join(test_dirc, sd_sgd_sol), os.path.join(test_dirc, sd_init)
    hessian_path = os.path.join(test_dirc, hessian_file)
    print(hessian_path)
    HPB = PacBayes_Hessian_partial(net, datasets, criterion_nn, accuracy_loss)
    hessian_data = torch.load(hessian_path)
    eigenvals = torch.Tensor(hessian_data['eigenvals'])
    eigenvecs = torch.Tensor(hessian_data['eigenvecs'])
    _, indices = torch.sort(eigenvals.abs(), descending=True)
    indices = indices[:top_n]
    HPB.load_eigenthings(eigenvals[indices], eigenvecs[indices])
    HPB.load_sd(sd_path_sgd_sol)
    gap = HPB.generalization_gap()
    print(gap)
    HPB.evaluate(True, True)

    mean_prior = HPB.get_prior_mean(sd_path_init)
    HPB.initialize_BRE_eigenval(mean_prior)

    # HPB.optimize_PACB_RMSprop(learning_rate=0.001, epoch_num=3000, lr_decay_mode='step', lr_gamma=0.1, step_lr_decay=1000)
    HPB.optimize_PACB_RMSprop(learning_rate=0.001, epoch_num=500, lr_decay_mode='step', lr_gamma=0.1, step_lr_decay=300)
    # HPB.optimize_PACB_RMSprop(learning_rate=0.01, epoch_num=800, lr_decay_mode='exp', lr_gamma=0.1 ** (1/40))
    # exit()
    HPB.compute_bound(n_monte_carlo_approx=1000, sample_freq=100)
    print(list(HPB.BRE.sigma_post_.detach().to('cpu').numpy()))
    torch.save(HPB.BRE.sigma_post_.detach().to('cpu'), 'tmp_log/sigma_post/hessian_top_run_4.pth')

if __name__ == '__main__':
    main()
