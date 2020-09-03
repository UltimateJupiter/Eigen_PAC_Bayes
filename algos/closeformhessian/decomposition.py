import torch
import scipy.linalg as sla
import numpy as np
from copy import deepcopy
import itertools
import scipy.linalg as sla
import itertools
import datetime
from .utils import *
from .dp import *

def weight_load(sd, fc_seq, sample_out):
    
    fc_seq.reverse()
    class_count = sample_out.size()[-1]
    Ws = [sd[layer + '.weight'] for layer in fc_seq]
    assert class_count == Ws[0].size()[0], (class_count, Ws[0].size(), fc_seq)
    for i in range(len(Ws) - 1):
        assert Ws[i].size()[1] == Ws[i + 1].size()[0], "{}-{} weight size mismatch".format(fc_seq[i], fc_seq[i + 1])
    fc_seq.reverse()

    return Ws

class Decomp():

    y_classification_mode_list = ['softmax',
                 'binary_logistic_pn1',
                 'multi_logistic_pn1',
                 'binary_logistic_01']

    def x_comp(self, ife, layers, inputs, device, Ws=None, out_device=None, batch_sum=False, **kwargs):
        
        inputs = inputs.to(device)
        mid_out, _ = ife(inputs) 
        
        ret = {}
        for layer in layers:
            assert layer in ife.target_layers
            x = mid_out[layer][0]
            x = x.view(inputs.size()[0], -1).unsqueeze(-1)
            if batch_sum:
                x = x.sum(axis=0)
            if out_device is not None:
                x = x.to(out_device)
            ret[layer] = x
        return ret

    def p_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, y_confidence_scale=1, y_classification_mode='softmax', labels=None, **kwargs):
        
        assert y_classification_mode in self.y_classification_mode_list
        inputs = inputs.to(device)
        _, final_out = ife(inputs)

        if y_confidence_scale != 1:
            assert y_confidence_scale
            final_out.mul_(y_confidence_scale)
            # log('confidence - {}'.format(y_confidence_scale))

        if y_classification_mode == 'softmax':
            softmax = torch.nn.Softmax(dim=1).to(device)
            p = softmax(final_out)

        elif y_classification_mode == 'binary_logistic_pn1':
            assert labels is not None
            if len(labels.shape) == 1:
                labels_comp = labels.unsqueeze(-1)
            p = torch.Tensor.sigmoid(labels_comp.mul(final_out))
        
        elif y_classification_mode == 'binary_logistic_01' or y_classification_mode == 'multi_logistic_pn1':
            # TODO: finish if needed
            print("y_classification_mode {} not finished yet".format(y_classification_mode))

        if batch_sum:
            p = p.sum(axis=0)
        if out_device is not None:
            p = p.to(out_device)
        ret = {layer: p for layer in layers}
        return ret
    
    def c_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):

        inputs = inputs.to(device)
        mid_out, final_out = ife(inputs)
        target_layers = deepcopy(ife.target_layers)
        target_layers.reverse()

        cs = []
        for layer in target_layers:
            _, feat_out = mid_out[layer][0], mid_out[layer][1]
            cs.append((feat_out >= 0).float())
        
        ret = {}
        for layer in layers:
            assert layer in target_layers
            i = target_layers.index(layer)
            c = cs[i]
            if batch_sum:
                c = c.sum(axis=0)
            if out_device is not None:
                c = c.to(out_device)
            ret[layer] = c

        empty_cache(device)
        return ret

    def C_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):

        inputs = inputs.to(device)
        mid_out, final_out = ife(inputs)
        target_layers = deepcopy(ife.target_layers)
        target_layers.reverse()

        Cs = []
        for layer in target_layers:
            _, feat_out = mid_out[layer][0], mid_out[layer][1]
            Cs.append(matrix_diag((feat_out >= 0).float()))
        
        ret = {}
        for layer in layers:
            assert layer in target_layers
            i = target_layers.index(layer)
            C = Cs[i]
            if batch_sum:
                C = C.sum(axis=0)
            if out_device is not None:
                C = C.to(out_device)
            ret[layer] = C

        empty_cache(device)
        return ret

    def U_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):

        inputs = inputs.to(device)
        softmax = torch.nn.Softmax(dim=1).to(device)
        mid_out, final_out = ife(inputs)
        target_layers = deepcopy(ife.target_layers)
        target_layers.reverse()

        Cs = []
        for layer in target_layers:
            _, feat_out = mid_out[layer][0], mid_out[layer][1]
            Cs.append(matrix_diag((feat_out >= 0).float()))
        
        Us = []
        batch_identity = torch.eye(final_out.size()[-1]).unsqueeze(0).repeat(final_out.size()[0], 1, 1).to(device) # pylint: disable=no-member
        Us.append(batch_identity)
        for i in range(len(target_layers) - 1):
            U_prev = Us[i]
            U_next = U_prev.matmul(Ws[i]).matmul(Cs[i + 1])
            Us.append(U_next)
        
        ret = {}
        for layer in layers:
            assert layer in target_layers
            i = target_layers.index(layer)
            U = Us[i]
            if batch_sum:
                U = U.sum(axis=0)
            if out_device is not None:
                U = U.to(out_device)
            ret[layer] = U

        empty_cache(device)
        return ret
    
    def xxT_comp(self, ife, layers, inputs, device, Ws=None, out_device=None, batch_sum=False, **kwargs):
        
        inputs = inputs.to(device)
        mid_out, _ = ife(inputs)
        
        ret = {}
        for layer in layers:
            assert layer in ife.target_layers
            feat_in, _ = mid_out[layer][0], mid_out[layer][1]
            feat_in = feat_in.view(inputs.size()[0], -1).unsqueeze(1)
            xxTs = torch.matmul(feat_in.transpose(1, 2), feat_in) # pylint: disable=no-member
            if batch_sum:
                xxTs = xxTs.sum(axis=0)
            if out_device is not None:
                xxTs = xxTs.to(out_device)
            ret[layer] = xxTs
        return ret

    def UTAU_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):

        Us = self.U_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)
        A = self.A_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)[layers[0]]
        
        ret = {}
        for layer in layers:
            UTAU = Us[layer].transpose(1, 2).matmul(A).matmul(Us[layer])
            if batch_sum:
                UTAU = UTAU.sum(axis=0)
            if out_device is not None:
                UTAU = UTAU.to(out_device)
            ret[layer] = UTAU

        empty_cache(device)
        return ret


    def H_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):
        UTAUs = self.UTAU_comp(ife, layers, inputs, device, Ws, **kwargs)
        xxTs = self.xxT_comp(ife, layers, inputs, device, **kwargs)
        ret = {}
        for layer in layers:
            H = bkp_2d(UTAUs[layer], xxTs[layer])
            if batch_sum:
                H = H.sum(axis=0)
            if out_device is not None:
                H = H.to(out_device)
            ret[layer] = H
        empty_cache(device)
        return ret

    def H_full_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, labels=None, **kwargs):
        inputs = inputs.to(device)
        labels = labels.to(device)
        _, final_out = ife(inputs)
        final_out = final_out.view(-1)
        target_layers = deepcopy(ife.target_layers)
        p = torch.Tensor.sigmoid(labels.mul(final_out))
        A = p.mul(1 - p)
        A = A.div_(np.log(2))
        A = A.unsqueeze(-1)
        Us = self.U_comp(ife, target_layers, inputs, device, Ws)
        xs = self.x_comp(ife, target_layers, inputs, device)
        M = []
        for layer in target_layers:
            M.append(bkp_2d(Us[layer].transpose(1,2), xs[layer]))
            M.append(Us[layer].transpose(1,2))
        M = torch.cat(M, dim=1)
        H = M.mul(A)
        H = H.matmul(M.transpose_(1,2))
        if batch_sum:
            H = H.sum(axis=0)
        if out_device is not None:
            H = H.to(out_device)
        empty_cache(device)
        ret = {layer: H for layer in layers}
        return ret

    def L_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):

        A = self.A_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)[layers[0]]
        A = A.to('cpu').double()
        L = torch.empty_like(A) # pylint: disable=no-member
        for i in range(A.shape[0]):
            Lu, D, _ = sla.ldl(A[i].numpy())
            D = np.maximum(D, 0) # pylint: disable=assignment-from-no-return
            Lu = np.matmul(Lu, np.sqrt(D))
            L[i] = torch.from_numpy(Lu) # pylint: disable=no-member
        A = A.float().to(device)
        L = L.float().to(device)
        if batch_sum:
            L = L.sum(axis=0)
        if out_device is not None:
            L = L.to(out_device)
        ret = {layer: L for layer in layers}
        return ret

    def A_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, y_classification_mode='softmax', **kwargs):

        p = self.p_comp(ife, layers, inputs, device, Ws, out_device=device, y_classification_mode=y_classification_mode, **kwargs)[layers[0]]
        
        if y_classification_mode == 'softmax':
            diag_p = matrix_diag(p)
            p_mat = p.unsqueeze(1)
            ppTs = torch.matmul(p_mat.transpose(1, 2), p_mat) # pylint: disable=no-member
            A = diag_p - ppTs

        elif y_classification_mode == 'binary_logistic_pn1':
            A = p.mul(1 - p)
            A.div_(np.log(2))
        
        elif y_classification_mode == 'binary_logistic_01' or y_classification_mode == 'multi_logistic_pn1':
            # TODO: finish if needed
            print("y_classification_mode {} not finished yet".format(y_classification_mode))
        
        if batch_sum:
            A = A.sum(axis=0)
        if out_device is not None:
            A = A.to(out_device)
        ret = {layer: A for layer in layers}
        return ret

    def Ah_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):

        A = self.A_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)[layers[0]]
        vals, vecs = eigenthings_tensor_utils_batch(A, symmetric=True, out_device=device)
        assert torch.min(vals) > -1e-6 # pylint: disable=no-member
        vals *= vals > 0
        vals_sqrt = torch.sqrt(vals) # pylint: disable=no-member
        D = matrix_diag(vals_sqrt)
        Ah = torch.bmm(torch.bmm(vecs.transpose(1, 2), D), vecs) # pylint: disable=no-member
        if batch_sum:
            Ah = Ah.sum(axis=0)
        if out_device is not None:
            Ah = Ah.to(out_device)
        ret = {layer: Ah for layer in layers}
        return ret

    def UxT_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):
        x = self.x_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)
        Us = self.U_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)
        
        ret = {}
        for layer in layers:
            UxT = bkp_2d(Us[layer], x[layer].transpose(-1, -2))
            if batch_sum:
                UxT = UxT.sum(axis=0)
            if out_device is not None:
                UxT = UxT.to(out_device)
            ret[layer] = UxT
        empty_cache(device)
        return ret

    def UxT_norm_comp(self, ife, layers, inputs, device, Ws, out_device=None, batch_sum=False, **kwargs):
        x = self.x_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)
        Us = self.U_comp(ife, layers, inputs, device, Ws, out_device=device, **kwargs)
        
        ret = {}
        for layer in layers:
            UxT = bkp_2d(Us[layer], x[layer])
            UxT = UxT.div_(UxT.norm(dim=-1).unsqueeze(-1))
            if batch_sum:
                UxT = UxT.sum(axis=0)
            if out_device is not None:
                UxT = UxT.to(out_device)
            ret[layer] = UxT
        empty_cache(device)
        return ret

    def eigenthings_exp_hessian_approx(self, E_UTAU, E_xxT, num_eigenthings, device, out_device, comp_vectors=True, timer_on=False, symmetric_exact=False):
        
        E_UTAU_ondevice, E_xxT_ondevice = E_UTAU.to(device), E_xxT.to(device)

        if symmetric_exact:
            eigenvals_UTAU, eigenvecs_UTAU = eigenthings_tensor_utils(E_UTAU_ondevice, symmetric=True) # pylint: disable=no-member
        else:
            eigenvals_UTAU, eigenvecs_UTAU = eigenthings_tensor_utils(E_UTAU_ondevice, topn=num_eigenthings) # pylint: disable=no-member
        eigenvals_UTAU = eigenvals_UTAU.cpu().numpy()

        if symmetric_exact:
            eigenvals_xxT, eigenvecs_xxT = eigenthings_tensor_utils(E_xxT_ondevice, symmetric=True) # pylint: disable=no-member
        else:
            eigenvals_xxT, eigenvecs_xxT = eigenthings_tensor_utils(E_xxT_ondevice, topn=num_eigenthings) # pylint: disable=no-member
        eigenvals_xxT = eigenvals_xxT.cpu().numpy()
        kron_pairs, eigenvals = kmax_argsort(eigenvals_UTAU, eigenvals_xxT, num_eigenthings, return_vals=True)

        if not comp_vectors:
            return kron_pairs
        eigenvecs = []
        for p in kron_pairs:
            eigenvec_approx = eigenvecs_UTAU[p[0]].unsqueeze(1).matmul(eigenvecs_xxT[p[1]].unsqueeze(0)).view(-1)
            eigenvecs.append(eigenvec_approx.unsqueeze(0))
        eigenvecs = torch.cat(eigenvecs, axis=0).to(out_device) # pylint: disable=no-member

        return eigenvals, eigenvecs, kron_pairs
