import torch.utils.data as du
import numpy as np

from torchvision import datasets, transforms


class _CIFAR():
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda

    def data_def(self, args):
        args.input_channels = 3
        args.im_size = 32
        args.input_dim = 32*32*3
        args.output_dim = 10
        if args.dataset == 'CIFAR100':
            args.output_dim = 100

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.use_cuda else {}
        transform_train=transforms.Compose([
            transforms.RandomCrop(args.im_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        noise_radius = [((-(0-0.4914)/0.2023) + ((1-0.4914)/0.2023))*(args.attack_radius/255.0),
                        ((-(0-0.4822)/0.1994) + ((1-0.4822)/0.1994))*(args.attack_radius/255.0),
                        ((-(0-0.4465)/0.2010) + ((1-0.4465)/0.2010))*(args.attack_radius/255.0)]

        attack_step_size = [((-(0-0.4914)/0.2023) + ((1-0.4914)/0.2023))*(args.attack_stepsize/255.0),
                            ((-(0-0.4822)/0.1994) + ((1-0.4822)/0.1994))*(args.attack_stepsize/255.0),
                            ((-(0-0.4465)/0.2010) + ((1-0.4465)/0.2010))*(args.attack_stepsize/255.0)]

        pytorch_range = [[((0-0.4914)/0.2023), ((1-0.4914)/0.2023)],
                         [((0-0.4822)/0.1994), ((1-0.4822)/0.1994)],
                         [((0-0.4465)/0.2010), ((1-0.4465)/0.2010)]]

        if args.dataset == 'CIFAR10':
            train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
        elif args.dataset == 'CIFAR100':
            train_set = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
            test_set = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_test)
        else:
            print('Dataset type "{0}" not recognized, exiting ...'.format(args.dataset))
            exit()

        if args.val_set == 'TRAIN':
            train_frac = 0.9
        else:
            train_frac = 1
        num_train = len(train_set)
        indices = list(range(num_train))

        args.dataset_size = int(np.floor(train_frac * num_train))
        train_idx, val_idx = indices[:args.dataset_size], indices[args.dataset_size:]
        train_sampler = du.sampler.SubsetRandomSampler(train_idx)
        val_sampler = du.sampler.SubsetRandomSampler(val_idx)
        val_set = train_set

        return train_set, val_set, test_set, train_sampler, val_sampler, kwargs, pytorch_range, noise_radius, attack_step_size