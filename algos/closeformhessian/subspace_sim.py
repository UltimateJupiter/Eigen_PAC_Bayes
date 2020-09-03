import torch

def overlap_calc(subspace1, subspace2, device):
    M = subspace1.to(device)
    V = subspace2.to(device)
    Mt = torch.transpose(M, 0, 1) # pylint: disable=no-member
    n = M.size()[0]
    k = 0
    for i in range(n):
        vi = V[i]
        li = Mt.mv(M.mv(vi))
        ki = torch.dot(li, li) # pylint: disable=no-member
        k += ki
    del Mt, M, V
    torch.cuda.empty_cache()
    return (k / n).to('cpu').numpy()