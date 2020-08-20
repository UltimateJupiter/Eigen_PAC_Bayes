import sys, os
import torch
from hpb.hessianpacbayes import *

test_dirc = '../hessian_eigenspace_overlap/CIFAR10_Exp1/experiments/LeNet5_fixlr0.01'
sd_file = 'experiment_log/run_1/models/final.pth'
test_dirc = '../hessian_eigenspace_overlap/MNIST_Exp1/experiments/FC2_fixlr0.01'
test_dirc = '../hessian_eigenspace_overlap/MNIST_Binary/experiments/FC1_600_fixlr0.01'
def main():

    assert os.path.isdir(test_dirc)
    sys.path.append(test_dirc)
    from config import Config # pylint: disable=no-name-in-module
    conf = Config()

    net = conf.net()
    datasets = [conf.dataset(train=True, transform=conf.train_transform), conf.dataset(train=False, transform=conf.train_transform)]
    sd_path = os.path.join(test_dirc, sd_file)
    assert os.path.isfile(sd_path)

    HPB = PacBayes_Optim(net, datasets)
    HPB.load_sd(sd_path)
    gap = HPB.generalization_gap()
    HPB.evaluate(train=True, log=True)

    HPB.initialize_BRE()
    HPB.optimize_PACB_RMSprop()
    HPB.compute_bound()

if __name__ == '__main__':
    main()