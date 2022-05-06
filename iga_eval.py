""" quantized classifier with given model -- main script
"""

import matplotlib
matplotlib.use('Agg')
import logging
import os, random
import numpy as np
import utils.utils as util
import torch
import torch.utils.data as du
import cfgs.cfg as cfg_exp

from datetime import datetime
from datasets.CIFAR import _CIFAR
from datasets.MNIST import _MNIST
from quant_adversarial.CONTINUOUS import _Continuous_Nets
from quant_adversarial.BINARIZED_NET import _Binarized_Nets
from quant_adversarial.BNN_WAQ import _BNN_WAQ_Nets

logging.getLogger('matplotlib.font_manager').disabled = True

RANDOM_SEED = 123456

def seed_torch():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_fn(worker_id):
    random.seed(RANDOM_SEED + worker_id)
    np.random.seed(RANDOM_SEED + worker_id)


def main():
    # Get the CL arguments
    args = cfg_exp.args

    seed_torch()

    # pytorch setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.quant_levels == 3:
        if args.full_ste:
            args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method + '_TNN_STE', args.exp_name,
                                         args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method + '_TNN',
                                         args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    else:
        if args.full_ste:
            args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method + '_STE',
                                         args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            args.save_dir = os.path.join(args.save_dir, args.dataset, args.architecture, args.method,
                                         args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    args.save_name = os.path.join(args.save_dir, 'best_model.pth.tar')
    args.save_name_adv = os.path.join(args.save_dir, 'best_model_adv.pth.tar')

    util.setup_logging(os.path.join(args.save_dir, 'log.txt'))
    results_file = os.path.join(args.save_dir, 'results.%s')
    results = util.ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("Saving to %s", args.save_dir)

    # load data
    args.data_path = os.path.join(args.data_path, args.dataset)

    if 'CIFAR' in args.dataset:
        data_class = _CIFAR(use_cuda)
    elif args.dataset == 'MNIST':
        data_class = _MNIST(use_cuda)
    else:
        print('Dataset type "{0}" not recognized, exiting ...'.format(args.dataset))
        exit()

    train_set, val_set, test_set, train_sampler, val_sampler, \
    kwargs, pytorch_range, noise_radius, attack_step_size = data_class.data_def(args)

    test_loader = du.DataLoader(test_set, batch_size=args.batch_size, worker_init_fn=init_fn, **kwargs)

    if args.val_set == 'TEST':
        val_loader = test_loader

    print(args)
    logging.debug("Run arguments: %s", args)

    quant_methods = ['BNN', 'BNN_STE']

    if args.method == 'CONTINUOUS':
        method_class = _Continuous_Nets(args)
    elif args.method == 'BNN' or args.method == 'BNN_STE':
        method_class = _Binarized_Nets(args)
    elif args.method == 'BNN_WAQ':
        method_class = _BNN_WAQ_Nets(args)
    else:
        print('Method "{0}" not recognized, exiting ...'.format(args.method))
        exit()

    model, model_adv = method_class.model_def(device)

    if args.eval:
        model = method_class.load_model(model)

    if args.method in quant_methods:
        method_class.doround(model)

    test_loader_jac = du.DataLoader(test_set, batch_size=1, worker_init_fn=init_fn, **kwargs)

    if args.eval:
        # #### Evaluated adversarial accuracy on clean model
        method_class.evaluate_adv(model, device, test_loader, test_loader_jac, noise_radius, attack_step_size, pytorch_range,
                                  adv_training=False, random_restarts=args.random_restarts)

    ##### If using stored adv. trained model.
    if args.eval_adv:
        model_adv = method_class.load_model_adv(model_adv)

        if args.method in quant_methods:
            method_class.doround(model_adv)

        # #### Evaluate adversarial accuracy on adversarially trained model
        method_class.evaluate_adv(model_adv, device, test_loader, test_loader_jac, noise_radius, attack_step_size, pytorch_range,
                                  adv_training=True, random_restarts=args.random_restarts)


if __name__ == '__main__':
    main()
