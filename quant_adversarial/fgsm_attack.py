import torchvision.transforms as transforms
import torch.nn as nn
import torch
import numpy as np
from quant_adversarial.nonlinearity_beta import estimate_beta_nonlinearity_hessorig


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.data.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)


def fgsm(inputs, net, targets=None, pytorch_range=[], step_size=0.04, epsil=5./255.*8):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and
       perturbed image
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    pert_image = inputs.clone().to(device)

    if inputs.shape[1] == 3:
        pert_image[:, 0, :, :] = pert_image[:, 0, :, :] + (torch.rand(inputs.shape)[:, 0, :, :].to(device)-0.5)*2*epsil[0]
        pert_image[:, 1, :, :] = pert_image[:, 1, :, :] + (torch.rand(inputs.shape)[:, 1, :, :].to(device)-0.5)*2*epsil[1]
        pert_image[:, 2, :, :] = pert_image[:, 2, :, :] + (torch.rand(inputs.shape)[:, 2, :, :].to(device)-0.5)*2*epsil[2]
    else:
        pert_image[:, 0, :, :] = pert_image[:, 0, :, :] + (torch.rand(inputs.shape)[:, 0, :, :].to(device)-0.5)*2*epsil

    pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
    pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
    pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])

    pert_image = pert_image.to(device)
    pert_image.requires_grad_()
    zero_gradients(pert_image)
    fs = net.eval()(pert_image)
    loss_wrt_label = nn.CrossEntropyLoss()(fs, targets)
    grad = torch.autograd.grad(loss_wrt_label, pert_image, only_inputs=True, create_graph=False, retain_graph=False)[0]

    dr = torch.sign(grad.data)
    # dr = grad.data

    pert_image.detach_()
    if pert_image.shape[1] == 3:
        pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size[0]
        pert_image[:, 1, :, :] += dr[:, 1, :, :] * step_size[1]
        pert_image[:, 2, :, :] += dr[:, 2, :, :] * step_size[2]

        pert_image[:, 0, :, :] = torch.min(torch.max(pert_image[:, 0, :, :],
                                                     inputs[:, 0, :, :] - epsil[0]), inputs[:, 0, :, :] + epsil[0])
        pert_image[:, 1, :, :] = torch.min(torch.max(pert_image[:, 1, :, :],
                                                     inputs[:, 1, :, :] - epsil[1]), inputs[:, 1, :, :] + epsil[1])
        pert_image[:, 2, :, :] = torch.min(torch.max(pert_image[:, 2, :, :],
                                                     inputs[:, 2, :, :] - epsil[2]), inputs[:, 2, :, :] + epsil[2])

        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
        pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
        pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])
    else:
        pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size
        pert_image[:, 0, :, :] = torch.min(torch.max(pert_image[:, 0, :, :],
                                                     inputs[:, 0, :, :] - epsil), inputs[:, 0, :, :] + epsil)
        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0], pytorch_range[1])
    r_tot = pert_image - inputs
    return r_tot


def modified_fgsm(inputs, net, args, sp_scalar, targets=None, pytorch_range=[], step_size=0.04, epsil=5./255.*8):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    pert_image = inputs.clone().to(device)

    if inputs.shape[1] == 3:
        pert_image[:, 0, :, :] = pert_image[:, 0, :, :] + (torch.rand(inputs.shape)[:, 0, :, :].to(device)-0.5)*2*epsil[0]
        pert_image[:, 1, :, :] = pert_image[:, 1, :, :] + (torch.rand(inputs.shape)[:, 1, :, :].to(device)-0.5)*2*epsil[1]
        pert_image[:, 2, :, :] = pert_image[:, 2, :, :] + (torch.rand(inputs.shape)[:, 2, :, :].to(device)-0.5)*2*epsil[2]
    else:
        pert_image[:, 0, :, :] = pert_image[:, 0, :, :] + (torch.rand(inputs.shape)[:, 0, :, :].to(device)-0.5)*2*epsil

    pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
    pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
    pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])

    pert_image = pert_image.to(device)

    if args.modified_attack_technique=='Beta_NonLinearity_hessorig':
        beta_batch = estimate_beta_nonlinearity_hessorig(pert_image, targets, net.eval())

    pert_image.requires_grad_()
    zero_gradients(pert_image)
    fs = net.eval()(pert_image)
    if args.modified_attack_technique=='TS_gradthresh_MJSVnetwork':
        fs = fs/sp_scalar
        with torch.no_grad():
            sorted_fs, _ = torch.sort(fs, dim=1)
            gamma = sorted_fs[:, -1] - sorted_fs[:, 0]
            # epsilon = args.gradthresh_epsilon
            epsilon = 1e-2
            d = fs.shape[1]
            bs = fs.shape[0]
            ## for gt class  logits grad thresholding
            thresh1 = (-1/gamma)*np.log(epsilon/((1-epsilon)*(d-1)))
            thresh1 = thresh1.unsqueeze(1).repeat(1, d)
            softmax_fs = torch.softmax(fs, dim=1)
        gt_grad_bool = torch.abs(1.0 - softmax_fs[np.arange(0, bs), targets]) < epsilon
        scaled_fs = torch.where(gt_grad_bool.unsqueeze(1)==1, fs*thresh1, fs)
        loss_wrt_label = nn.CrossEntropyLoss()(scaled_fs, targets)
    if args.modified_attack_technique=='Beta_NonLinearity_hessorig':
        softmax_fs = torch.softmax(fs, dim=1)
        d = fs.shape[1]
        scaled_fs = fs*beta_batch.unsqueeze(1).repeat(1, d)
        loss_wrt_label = nn.CrossEntropyLoss()(scaled_fs, targets)
    grad = torch.autograd.grad(loss_wrt_label, pert_image, only_inputs=True, create_graph=False, retain_graph=False)[0]

    dr = torch.sign(grad.data)
    # dr = grad.data

    pert_image.detach_()
    if pert_image.shape[1] == 3:
        pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size[0]
        pert_image[:, 1, :, :] += dr[:, 1, :, :] * step_size[1]
        pert_image[:, 2, :, :] += dr[:, 2, :, :] * step_size[2]

        pert_image[:, 0, :, :] = torch.min(torch.max(pert_image[:, 0, :, :],
                                                     inputs[:, 0, :, :] - epsil[0]), inputs[:, 0, :, :] + epsil[0])
        pert_image[:, 1, :, :] = torch.min(torch.max(pert_image[:, 1, :, :],
                                                     inputs[:, 1, :, :] - epsil[1]), inputs[:, 1, :, :] + epsil[1])
        pert_image[:, 2, :, :] = torch.min(torch.max(pert_image[:, 2, :, :],
                                                     inputs[:, 2, :, :] - epsil[2]), inputs[:, 2, :, :] + epsil[2])

        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
        pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
        pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])
    else:
        pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size
        pert_image[:, 0, :, :] = torch.min(torch.max(pert_image[:, 0, :, :],
                                                     inputs[:, 0, :, :] - epsil), inputs[:, 0, :, :] + epsil)
        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0], pytorch_range[1])
    r_tot = pert_image - inputs
    return r_tot