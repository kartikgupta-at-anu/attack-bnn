import torch
import torch.nn as nn
import numpy as np
from utils.visualise_adv import jacobian_faster, softmaxjacobian_faster
import sys


def estimate_beta_nonlinearity_hessorig(inputs, targets, net):
    beta_batch = torch.zeros(inputs.shape[0], device='cuda')
    for i in range(inputs.shape[0]):
        curr_input = inputs[i:i+1]
        label = targets[i:i+1]
        # curr_input.requires_grad_()
        with torch.no_grad():
            out = net(curr_input)
        out.requires_grad_()
        loss = nn.CrossEntropyLoss()
        beta = 1.0
        scaled_out = beta*out
        soft_scaled = torch.softmax(scaled_out, dim=1)
        if not(torch.argmax(soft_scaled, dim=1) == label):
            beta_batch[i] = 1.0
            continue;
        l = loss(scaled_out, label)
        jac = jacobian_faster(net.eval(), curr_input, out.shape[1])

        localgrad_softmax = softmaxjacobian_faster(scaled_out)

        a2 = torch.matmul(beta*localgrad_softmax,jac)
        a2 = torch.matmul(a2.permute(1, 0), jac)
        hess_oureqn = beta*(a2)


        max_hessnorm = 0
        max_hessnorm_beta = 0
        norms = []
        betas = []

        ## for gt class  logits grad thresholding
        d = out.shape[1]
        epsilon = (1 - 1.0/d - 1e-2)
        sorted_out, _ = torch.sort(out)
        gamma = sorted_out[0, -1] - sorted_out[0, -2]
        beta1 = (-1/gamma)*np.log(epsilon/((1-epsilon)*(d-1)))
        epsilon = 1e-72
        beta2 = (-1/gamma)*np.log(epsilon/((1-epsilon)*(d-1)))

        if beta1<=0 or beta2<=0:
            sys.exit('You gave me negative Betas. WTF !!!!')

        beta_range = np.linspace(beta1.cpu().data.numpy(), beta2.cpu().data.numpy(), 100, endpoint=True)
        for beta in beta_range:
            scaled_out = beta*out
            soft_scaled = torch.softmax(scaled_out, dim=1)
            l = loss(scaled_out, label)
            localgrad_softmax = softmaxjacobian_faster(scaled_out)
            a2 = torch.matmul(beta*localgrad_softmax,jac)
            a2 = torch.matmul(a2.permute(1, 0), jac)
            hess_oureqn = beta*(a2)

            if hess_oureqn.norm() > max_hessnorm:
                max_hessnorm = hess_oureqn.norm()
                max_hessnorm_beta = beta

            norms.append(hess_oureqn.norm().cpu().data.numpy())
            betas.append(beta)

        if (np.argmax(norms) == 0) or (np.argmax(norms) == len(betas)-1):
            max_hessnorm_beta = 1.0
        beta_batch[i] = max_hessnorm_beta
    return beta_batch
