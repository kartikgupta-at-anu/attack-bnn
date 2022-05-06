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


def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)


def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)


def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()


def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def pgd_l2(inputs, net, epsilon=[1.], targets=None, pytorch_range=[], step_size=0.04, num_steps=20, epsil=5./255.*8):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and
       perturbed image
    """
    step_size = 2.5*(np.array(epsil)/num_steps)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with torch.no_grad():
        pert_image = inputs.clone().to(device)
        delta = inputs.clone().to(device)

        if inputs.shape[1] == 3:
            delta.data[:, 0, :, :].uniform_(pytorch_range[0][0], pytorch_range[0][1])
            delta.data[:, 1, :, :].uniform_(pytorch_range[1][0], pytorch_range[1][1])
            delta.data[:, 2, :, :].uniform_(pytorch_range[2][0], pytorch_range[2][1])
            delta.data = delta.data - pert_image
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil[0])
            delta.data[:, 1, :, :] = clamp_by_pnorm(delta.data[:, 1, :, :], 2, epsil[1])
            delta.data[:, 2, :, :] = clamp_by_pnorm(delta.data[:, 2, :, :], 2, epsil[2])
        else:
            delta.data[:, 0, :, :].uniform_(pytorch_range[0], pytorch_range[1])
            delta.data = delta.data - pert_image
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil)

        x_adv = pert_image + delta
        x_adv[:, 0, :, :] = torch.clamp(x_adv[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
        x_adv[:, 1, :, :] = torch.clamp(x_adv[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
        x_adv[:, 2, :, :] = torch.clamp(x_adv[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])
        delta = x_adv - pert_image
        delta = delta.to(device)
        pert_image = pert_image.to(device)

    for ii in range(num_steps):
        delta.requires_grad_()
        zero_gradients(pert_image)
        zero_gradients(delta)
        fs = net.eval()(pert_image + delta)
        loss_wrt_label = nn.CrossEntropyLoss()(fs, targets)
        grad = torch.autograd.grad(loss_wrt_label, delta, only_inputs=True, create_graph=False, retain_graph=False)[0]
        dr = grad.data
        delta.detach_()
        dr = normalize_by_pnorm(dr)
        pert_image.detach_()
        if pert_image.shape[1] == 3:
            pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size[0]
            pert_image[:, 1, :, :] += dr[:, 1, :, :] * step_size[1]
            pert_image[:, 2, :, :] += dr[:, 2, :, :] * step_size[2]

            pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
            pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
            pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])

            delta = pert_image - inputs
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil[0])
            delta.data[:, 1, :, :] = clamp_by_pnorm(delta.data[:, 1, :, :], 2, epsil[1])
            delta.data[:, 2, :, :] = clamp_by_pnorm(delta.data[:, 2, :, :], 2, epsil[2])
        else:
            pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size
            pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0], pytorch_range[1])
            delta = pert_image - inputs
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil)

    pert_image = inputs + delta
    if pert_image.shape[1] == 3:
        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
        pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
        pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])
    else:
        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0], pytorch_range[1])
    r_tot = pert_image - inputs
    return r_tot


def modified_pgd_l2(inputs, net, args, sp_scalar, epsilon=[1.], targets=None, pytorch_range=[], step_size=0.04, num_steps=20, epsil=5./255.*8):

    step_size = 2.5*(np.array(epsil)/num_steps)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with torch.no_grad():
        pert_image = inputs.clone().to(device)
        delta = inputs.clone().to(device)

        if inputs.shape[1] == 3:
            delta.data[:, 0, :, :].uniform_(pytorch_range[0][0], pytorch_range[0][1])
            delta.data[:, 1, :, :].uniform_(pytorch_range[1][0], pytorch_range[1][1])
            delta.data[:, 2, :, :].uniform_(pytorch_range[2][0], pytorch_range[2][1])
            delta.data = delta.data - pert_image
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil[0])
            delta.data[:, 1, :, :] = clamp_by_pnorm(delta.data[:, 1, :, :], 2, epsil[1])
            delta.data[:, 2, :, :] = clamp_by_pnorm(delta.data[:, 2, :, :], 2, epsil[2])
        else:
            delta.data[:, 0, :, :].uniform_(pytorch_range[0], pytorch_range[1])
            delta.data = delta.data - pert_image
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil)

        x_adv = pert_image + delta
        x_adv[:, 0, :, :] = torch.clamp(x_adv[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
        x_adv[:, 1, :, :] = torch.clamp(x_adv[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
        x_adv[:, 2, :, :] = torch.clamp(x_adv[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])
        delta = x_adv - pert_image
        delta = delta.to(device)
        pert_image = pert_image.to(device)

    if args.modified_attack_technique=='Beta_NonLinearity_hessorig':
        beta_batch = estimate_beta_nonlinearity_hessorig(pert_image, targets, net.eval())

    for ii in range(num_steps):
        delta.requires_grad_()
        zero_gradients(pert_image)
        zero_gradients(delta)
        fs = net.eval()(pert_image + delta)
        if args.modified_attack_technique=='Beta_NonLinearity_hessorig':
            softmax_fs = torch.softmax(fs, dim=1)
            d = fs.shape[1]
            scaled_fs = fs*beta_batch.unsqueeze(1).repeat(1, d)
            loss_wrt_label = nn.CrossEntropyLoss()(scaled_fs, targets)
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
        grad = torch.autograd.grad(loss_wrt_label, delta, only_inputs=True, create_graph=False, retain_graph=False)[0]
        dr = grad.data
        delta.detach_()
        dr = normalize_by_pnorm(dr)
        pert_image.detach_()
        if pert_image.shape[1] == 3:
            pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size[0]
            pert_image[:, 1, :, :] += dr[:, 1, :, :] * step_size[1]
            pert_image[:, 2, :, :] += dr[:, 2, :, :] * step_size[2]

            pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
            pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
            pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])

            delta = pert_image - inputs
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil[0])
            delta.data[:, 1, :, :] = clamp_by_pnorm(delta.data[:, 1, :, :], 2, epsil[1])
            delta.data[:, 2, :, :] = clamp_by_pnorm(delta.data[:, 2, :, :], 2, epsil[2])
        else:
            pert_image[:, 0, :, :] += dr[:, 0, :, :] * step_size
            pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0], pytorch_range[1])
            delta = pert_image - inputs
            delta.data[:, 0, :, :] = clamp_by_pnorm(delta.data[:, 0, :, :], 2, epsil)

    pert_image = inputs + delta
    if pert_image.shape[1] == 3:
        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0][0], pytorch_range[0][1])
        pert_image[:, 1, :, :] = torch.clamp(pert_image[:, 1, :, :], pytorch_range[1][0], pytorch_range[1][1])
        pert_image[:, 2, :, :] = torch.clamp(pert_image[:, 2, :, :], pytorch_range[2][0], pytorch_range[2][1])
    else:
        pert_image[:, 0, :, :] = torch.clamp(pert_image[:, 0, :, :], pytorch_range[0], pytorch_range[1])
    r_tot = pert_image - inputs
    return r_tot