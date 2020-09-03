import sys, os
import torch
from hpb.hessianpacbayes import *

sd_sgd_sol = 'experiment_log/run_1/models/final.pth'
sd_init = 'experiment_log/run_1/models/epoch0.pth'
save_path = 'tmp_log/sigma_post/hessian_approx_10class_FC2.pth'

test_dirc = '../hessian_eigenspace_overlap/MNIST_Exp1/experiments/FC2_fixlr0.01'
hessian_file = 'experiment_log/run_1/MNIST_Exp1_FC2_fixlr0.01_E-1_UTAU_xxT.eval'
test_dirc = '../hessian_eigenspace_overlap/MNIST_Exp1/experiments/FC1_600_fixlr0.01'
hessian_file = 'experiment_log/run_1/MNIST_Exp1_FC1_600_fixlr0.01_E-1_UTAU_xxT.eval'
test_dirc = '../hessian_eigenspace_overlap/MNIST_Exp1/experiments/FC2_600_fixlr0.01'
hessian_file = 'experiment_log/run_1/MNIST_Exp1_FC2_600_fixlr0.01_E-1_UTAU_xxT.eval'

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
    HPB = PacBayes_Hessian_approx(net, datasets, criterion_nn, accuracy_loss)
    HPB.load_hessian_file(hessian_path)
    HPB.load_sd(sd_path_sgd_sol)
    gap = HPB.generalization_gap()
    print(gap)
    HPB.evaluate(True, True)

    mean_prior = HPB.get_prior_mean(sd_path_init)
    HPB.initialize_BRE(mean_prior)

    # HPB.optimize_PACB_RMSprop(learning_rate=0.001, epoch_num=3000, lr_decay_mode='step', lr_gamma=0.1, step_lr_decay=1000)
    HPB.optimize_PACB_RMSprop(learning_rate=0.001, epoch_num=1000, lr_decay_mode='step', lr_gamma=0.1, step_lr_decay=200)
    # HPB.optimize_PACB_RMSprop(learning_rate=0.01, epoch_num=800, lr_decay_mode='exp', lr_gamma=0.1 ** (1/40))
    # exit()
    HPB.compute_bound(n_monte_carlo_approx=40000, sample_freq=100)
    #print(list(HPB.BRE.sigma_post_.detach().to('cpu').numpy()))
    torch.save(HPB.BRE.sigma_post_.detach().to('cpu'), save_path)

if __name__ == '__main__':
    main()
