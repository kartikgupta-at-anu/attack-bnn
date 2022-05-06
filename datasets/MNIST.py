import torch.utils.data as du
import numpy as np

from torchvision import datasets, transforms


class _MNIST():
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda

    def data_def(self, args):
        args.input_channels = 1
        args.im_size = 28
        args.input_dim = 28*28*1
        args.output_dim = 10

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        noise_radius = ((-(0-0.1307)/0.3081) + ((1-0.1307)/0.3081))*(args.attack_radius/255)
        attack_step_size = ((-(0-0.1307)/0.3081) + ((1-0.1307)/0.3081))*(args.attack_stepsize/255)
        pytorch_range = [((0-0.1307)/0.3081), ((1-0.1307)/0.3081)]

        train_set = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(args.data_path, train=False, download=True, transform=transform)

        train_frac = 5./6
        num_train = len(train_set)
        indices = list(range(num_train))

        args.dataset_size = int(np.floor(train_frac * num_train))
        train_idx, val_idx = indices[:args.dataset_size], indices[args.dataset_size:]
        train_sampler = du.sampler.SubsetRandomSampler(train_idx)
        val_sampler = du.sampler.SubsetRandomSampler(val_idx)
        val_set = train_set

        return train_set, val_set, test_set, train_sampler, val_sampler, kwargs, pytorch_range, noise_radius, attack_step_size