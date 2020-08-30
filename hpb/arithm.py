
import torch
import torch.nn as nn
from torch.nn.utils import *
from math import log, pi
import numpy as np
from scipy import optimize
import matplotlib
import matplotlib.pyplot as plt

def calc_kullback_leibler(lambda_prior, sigma_post, mean_prior, mean_post, d_size):
    """
    Explicit calculation of KL divergence between prior N(0, lambda_prior * Id) and posterior N(flat_mean_prior, sigma_posterior_)
    """
    # print('\nCalling kl')
    # print("lambda_prior, sigma_post")
    # print(lambda_prior, sigma_post)
    # print("lambda_prior, sigma_post - Grad")
    # print(lambda_prior.grad, sigma_post.grad)
    tr = torch.Tensor.norm(torch.Tensor.exp(2 * sigma_post), p=1) / lambda_prior # pylint:disable=no-member
    l2 = torch.Tensor.pow(torch.Tensor.norm(mean_post - mean_prior, p=2), 2) / lambda_prior
    # print("tr, l2")
    # print(tr, l2)
    
    d = d_size
    logdet_prior = d * torch.Tensor.log(lambda_prior)
    logdet_post = 2 * torch.Tensor.sum(sigma_post)
    kl = (tr + l2 - d + logdet_prior - logdet_post ) / 2.
    # print("kl")
    # print(kl)

    return kl

def calc_kl_partial(lambda_prior, sigma_post, mean_prior, mean_post, d_size):
    """
    Explicit calculation of KL divergence between prior N(0, lambda_prior * Id) and posterior N(flat_mean_prior, sigma_posterior_)
    """
    # print('\nCalling kl')
    # print("lambda_prior, sigma_post")
    # print(lambda_prior, sigma_post)
    # print("lambda_prior, sigma_post - Grad")
    # print(lambda_prior.grad, sigma_post.grad)
    tr = torch.norm(torch.exp(2 * sigma_post[:-1]), p=1).add(torch.exp(2* sigma_post[-1])) / lambda_prior # pylint:disable=no-member
    l2 = torch.Tensor.pow(torch.Tensor.norm(mean_post - mean_prior, p=2), 2) / lambda_prior
    # print("tr, l2")
    # print(tr, l2)
    
    d = d_size
    logdet_prior = d * torch.Tensor.log(lambda_prior)
    logdet_post = 2 * torch.Tensor.sum(sigma_post)
    kl = (tr + l2 - d + logdet_prior - logdet_post ) / 2.
    # print("kl")
    # print(kl)

    return kl


def calc_BRE_term(Precision, conf_param, bound, mean_prior, mean_post, lambda_prior_, sigma_posterior_, data_size, d_size): 
    """
    Explicit Calculation of the second term of the optimization problem (BRE)
    """

    lambda_prior = torch.Tensor.clamp(torch.Tensor.exp(2 * lambda_prior_ ), min = 1e-38, max = bound - 1e-8)
    kl = calc_kullback_leibler(lambda_prior, sigma_posterior_, mean_prior, mean_post, d_size)
    log_log = 2 * torch.Tensor.log(Precision * (torch.Tensor.log(bound / lambda_prior)))
    m = data_size
    log_ = log((((pi ** 2) * m) / (6 * conf_param)))
    #print(kl + log_log + log_)
    bre = torch.Tensor.sqrt((kl + log_log + log_) / (2 * (m - 1)))
    if torch.isnan(bre): # pylint: disable=no-member
        print(lambda_prior, kl, log_log, log_, bre)
        exit()
    return bre, kl



def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([q * log(q / p) if q > 0. else 0. for q, p in zip(Q, P)])


def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.-q], [p, 1.-p])


def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0 - 1e-9) <= 0.0:
        return 1.0 - 1e-9
    else:
        return optimize.brentq(f, q, 1.0 - 1e-9)


def train_model(num_epochs, loader, nn_model, criterion, optimizer, device):
    # TODO: Remove, or reimplement in hessianpacbayes.py

    MESSAGE = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    total_step = len(loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loader):
            # move tensors to the configured device
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # forward pass
            outputs = nn_model(images)
            loss = criterion(outputs.float(), labels.long())

            # backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(MESSAGE.format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

def test_model(loader, nn_model, device):

    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, labels in iter(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = nn_model(inputs)

            _, predicted = torch.Tensor.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels.long()).sum().item()

        print('Accuracy of the network on the testset: {} %'.format(100 * correct / total))
        
    
def apply_weights(model, modified_parameters, net_mean_prior):
    """
    Modify the parameters of a neural network 
    """
    indi = 0
    for name, ind, shape_ in net_mean_prior:
        model.state_dict()[name].data.copy_(modified_parameters[indi: indi + ind].view(shape_)) 
        indi += ind
    return(model)

