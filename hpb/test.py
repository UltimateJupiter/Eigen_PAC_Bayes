import torch
import pynvml

def eval_model(network, dataloader, criterion, accuracy_loss, device):

    network.zero_grad()
    network.eval()
    loss = 0.0
    corrected, total = 0, 0

    for data in iter(dataloader):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = network(inputs)
        loss += criterion(outputs, labels).item()
        correct_batch, total_batch = accuracy_loss(outputs, labels)
        corrected += correct_batch
        total += total_batch
    
    network.train()
    return {'loss': loss / len(dataloader), 'acc': 1. - corrected / total}

def test_error(network, dataloader, accuracy_loss, device):

    network.zero_grad()
    network.eval()
    corrected, total = 0, 0

    for data in iter(dataloader):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = network(inputs)
        correct_batch, total_batch = accuracy_loss(outputs, labels)
        corrected += correct_batch
        total += total_batch
    
    network.train()
    error = 1. - corrected / total
    return error
