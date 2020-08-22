import sys, os
import torch
from hpb.hessianpacbayes import *

sd_sgd_sol = 'experiment_log/run_1/models/final.pth'
sd_init = 'experiment_log/run_1/models/epoch0.pth'

# test_dirc = '../hessian_eigenspace_overlap/CIFAR10_Exp1/experiments/LeNet5_fixlr0.01'
# test_dirc = '../hessian_eigenspace_overlap/MNIST_Exp1/experiments/FC2_fixlr0.01'
test_dirc = '../hessian_eigenspace_overlap/MNIST_Binary/experiments/FC1_600_fixlr0.01_RL'
test_dirc = '../hessian_eigenspace_overlap/MNIST_Binary/experiments/FC2_600_fixlr0.01'
test_dirc = '../hessian_eigenspace_overlap/MNIST_Binary/experiments/FC1_600_fixlr0.01'

def main():

    assert os.path.isdir(test_dirc)
    sys.path.append(test_dirc)
    from config import Config # pylint: disable=no-name-in-module
    conf = Config()

    net = conf.net()
    datasets = [conf.dataset(train=True, transform=conf.train_transform), conf.dataset(train=False, transform=conf.train_transform)]
    sd_path_sgd_sol, sd_path_init = os.path.join(test_dirc, sd_sgd_sol), os.path.join(test_dirc, sd_init)

    HPB = PacBayes_Optim(net, datasets)
    HPB.load_sd(sd_path_sgd_sol)
    gap = HPB.generalization_gap()
    print(gap)

    mean_prior = HPB.get_prior_mean(sd_path_init)
    HPB.initialize_BRE(mean_prior=mean_prior)

    HPB.optimize_PACB_RMSprop(learning_rate=0.001, epoch_num=400, lr_decay_mode='step', lr_gamma=0.1, step_lr_decay=300)
    # HPB.optimize_PACB_RMSprop(learning_rate=0.01, epoch_num=800, lr_decay_mode='exp', lr_gamma=0.1 ** (1/40))
    # exit()
    HPB.compute_bound(n_monte_carlo_approx=1000, sample_freq=100)

if __name__ == '__main__':
    main()