
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
    tr = torch.Tensor.norm(torch.Tensor.exp(2 * sigma_post), p=1) / lambda_prior # pylint:disable=no-member
    l2 = torch.Tensor.pow(torch.Tensor.norm(mean_prior - mean_post, p=2), 2) / lambda_prior
    
    d = d_size
    logdet_prior = d * torch.Tensor.log(lambda_prior)
    logdet_post = 2 * torch.Tensor.sum(sigma_post)
    kl = (tr + l2 - d + logdet_prior - logdet_post ) / 2.

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
    bre = torch.Tensor.sqrt((kl + log_log + log_) / (2 * (m - 1)))
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
        
def run(model, network ,train_loader, test_loader, LEARNING_RATE, MOMENTUM, NUM_EPOCHS, device):
    # TODO: Remove or reimplent along with train_model
    # nn model
    nn_model = network.to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=LEARNING_RATE , momentum=MOMENTUM)

    # Train phase
    train_model(loader=train_loader, num_epochs=NUM_EPOCHS, nn_model=nn_model, criterion=criterion, optimizer=optimizer, device=device)

    # then test
    test_model(loader=test_loader, nn_model=nn_model, device=device)

    # finally save nn model
    torch.save(nn_model.state_dict(), 'SGD_solutions/%s.ckpt'%model)
    
def test_error(loader, nn_model, device):
    # TODO: Same as test_model
    """
    Compute the empirical error of neural network on a dataset loader
    """
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
        error = 1. - correct / total
        print(error)
        return(error)
    
    
def apply_weights(model, modified_parameters, net_mean_prior):
    """
    Modify the parameters of a neural network 
    """
    indi = 0
    for name, ind, shape_ in net_mean_prior:
        model.state_dict()[name].data.copy_(modified_parameters[indi: indi + ind].view(shape_)) 
        indi += ind
    return(model)


def print_weights(model):
    # TODO: Remain
    for name, weights in model.named_parameters():
        print(name)
        print(weights)


def plot_results(model_name, BRE_loss, Kl_value, NN_loss, norm_weights, norm_sigma, norm_lambda, epochs, initial_mean_prior):
    # TODO: Remain
    plt.style.use('ggplot')
    range_values = range(1, len(BRE_loss) + 1) 
    fig, axes = plt.subplots(6, 1, figsize=(18, 13))
    param_names = ['BRE Loss', 'NN Loss', 'KL-div', 'Weights_norm', 'Sigma_norm', 'Lambda_norm']
    param_data = [BRE_loss, NN_loss, Kl_value, norm_weights, norm_sigma, norm_lambda]
    plot_colors = ['green', 'blue', 'grey', 'red', 'yellow', 'black']

    for i in range(6):
        axes[i].plot(range_values, param_data[i], label=param_names[i], color=plot_colors[i])
        axes[i].set_ylabel(param_names[i])
        axes[i].set_xticks(range_values)
        if i > 0:
            min_y = min(param_data[i]).item()
            max_y = max(param_data[i]).item()
        else:
            min_y = min(param_data[i])
            max_y = max(param_data[i])
        
        axes[i].set_ylim(min_y, max_y)

        if i == 0:
            axes[i].set_title(str(model_name))
        elif i == 5:
            axes[i].set_xlabel('# Of Epochs')
    
    fig.legend()
    plt.tight_layout()
    plt.savefig('./final_results/' + str(model_name) + '_epochs_' + str(epochs) + '_wprior_' + str(initial_mean_prior))
    plt.plot()
