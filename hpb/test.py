import torch
import pynvml

def eval_model(network, dataloader, criterion, device):

    network.zero_grad()
    network.eval()
    loss = 0.0
    corrected, total = 0, 0

    for data in iter(dataloader):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = network(inputs)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.Tensor.max(outputs.data, 1)
        corrected += (predicted == labels).sum().item()
        total += labels.size(0)
    
    network.train()
    return {'loss': loss / len(dataloader), 'acc': corrected / total}