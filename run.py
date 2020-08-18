import sys, os
import torch
from hpb.hessianpacbayes import PacBayes_Naive

test_dirc = '../hessian_eigenspace_overlap/CIFAR10_Exp1/experiments/LeNet5_fixlr0.01'
sd_file = 'experiment_log/run_1/models/final.pth'

def main():

    assert os.path.isdir(test_dirc)
    sys.path.append(test_dirc)
    from config import Config # pylint: disable=no-name-in-module
    conf = Config()

    net = conf.net()
    datasets = [conf.dataset(train=True, transform=conf.train_transform), conf.dataset(train=False, transform=conf.train_transform)]
    sd_path = os.path.join(test_dirc, sd_file)
    assert os.path.isfile(sd_path)

    HPB = PacBayes_Naive(net, datasets)
    HPB.load_sd(sd_path)
    gap = HPB.generalization_gap()
    print(gap)
    HPB.evaluate(True, True)
    print('initialized')
    HPB.naive_bound()

if __name__ == '__main__':
    main()