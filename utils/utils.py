"""
Define some utility functions
"""
from copy import deepcopy
import pickle as pk
import os
import numpy as np
import subprocess
import sys

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader

import logging.config
import shutil
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
import cfgs.cfg as cfg

import torch

def read_vision_dataset(path, batch_size=128, num_workers=4, dataset='CIFAR10', transform=None):
    '''
    Read dataset available in torchvision

    Arguments:
        dataset : string
            The name of dataset, it should be available in torchvision
        transform_train : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
        transform_test : torchvision.transforms
            train image transformation
            if not given, the transformation for CIFAR10 is used
    Return:
        trainloader, testloader
    '''
    if not transform and dataset=='CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = getattr(datasets,dataset)(root=path, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = getattr(datasets,dataset)(root=path, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def get_random_dataset(dataset_size, input_dim, output_dim, one_hot = False):
    """ 
    Returns: 
        a dictionary with [dataset_size x input_dim] [dataset_size x 1]
    """
    dataset=dict()
    dataset['x'] = np.random.random_sample((dataset_size, input_dim)) # [0.0,1.0)
    tmp = np.random.randint(output_dim, size=dataset_size) # [0,output_dim)
    dataset['y'] = to_one_hot(tmp, dataset_size, output_dim) if one_hot else tmp
    return dataset

def save_obj(obj, name, save_dir):
    # Create directories to store the results
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    objfile = save_dir.rstrip('\/') + '/' + name + '.pkl'
    with open(objfile, 'wb') as f:
        pk.dump(obj, f, pk.HIGHEST_PROTOCOL)

def load_obj(name, save_dir):
    objfile = save_dir.rstrip('\/') + '/' + name + '.pkl'
    with open(objfile, 'rb') as f:
        return pk.load(f)

def to_one_hot(inp, length, labels):
    tmp = np.zeros((length, labels))
    tmp[np.arange(length), inp] = 1
    return tmp


# GEM, Arslan's code modifications!
def construct_split_mnist(task_labels, mnist_path='mnist.npz', one_hot=True):
    """
    Construct a split mnist dataset

    Args:
        task_labels     List of split labels

    Returns:
        dataset         A list of split datasets

    """
    # URL from: https://github.com/fchollet/keras/blob/master/keras/datasets/mnist.py
    if not os.path.exists(mnist_path):
        subprocess.call("wget https://s3.amazonaws.com/img-datasets/mnist.npz", shell=True)

    f = np.load('mnist.npz')
    mnist = {'train': dict(), 'test': dict(), 'val': dict()}
    # f['x_test'].shape[1] * f['x_test'].shape[2] == 784
    # f['x_train'].shape[0] == 60000
    mnist['train']['x'] = f['x_train'][:50000,:,:].reshape(50000, 784) / 255.
    mnist['train']['y'] = f['y_train'][:50000]
    mnist['val']['x'] = f['x_train'][50000:,:,:].reshape(10000, 784) / 255.
    mnist['val']['y'] = f['y_train'][50000:]
    mnist['test']['x'] = f['x_test'].reshape(f['x_test'].shape[0], 784) / 255.
    mnist['test']['y'] = f['y_test']

    #print(mnist['train']['x'].shape, mnist['train']['y'].shape, mnist['val']['x'].shape, mnist['val']['y'].shape)

    f.close()

    datasets = []

    sets = ['train', 'test', 'val']

    for task in task_labels:

        for set_name in sets:
            this_set = mnist[set_name]

            global_class_indices = np.column_stack((range(this_set['y'].shape[0]), this_set['y']))
            count = 0

            for cls in task:
                if count == 0:
                    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                    cls][:,np.array([True, False])])
                else:
                    class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] ==\
                                                                                             cls][:,np.array([True, False])]))
                count += 1

            class_indices = np.sort(class_indices, axis=None)
            
            mnist[set_name]['x'] = mnist[set_name]['x'][class_indices, :]
            tmp = mnist[set_name]['y'][class_indices]
            mnist[set_name]['y'] = to_one_hot(tmp, class_indices.shape[0], len(task)) if one_hot else tmp

        datasets.append(mnist)

    return datasets

def save_state(model, acc, args):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, args.save_name)

def create_val_folder(args):
    """
    create folder structure: root/class1/images.png
    only supports Tiny-Imagenet
    """
    assert(args.dataset == 'TINYIMAGENET200')

    path = os.path.join(args.data_path, 'val/images')  # path where validation data is present now
    filename = os.path.join(args.data_path, 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))

""" following is copied from BNN code 
"""
def accuracy(output, target, topk=(1,), avg=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        if avg:
            res.append(correct_k.mul_(100.0 / batch_size))
        else:
            res.append(correct_k)
    return res

def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    #def plot(self, *kargs, **kwargs):
    #    line = Line(data=self.results, *kargs, **kwargs)
    #    self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


def save_checkpoint(state, is_best, path='.', filename='checkpoint_adv.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'best_model_adv.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_adv_epoch_%s.pth.tar' % state['epoch']))


def save_model(state, filename):
    print('==> Saving model ...')
    torch.save(state, filename)


if cfg.args.fc_float and cfg.args.bias_float:
    def if_binary(n):
        return (not('bn' in n) and not('fc' in n) and not('bias' in n))
elif not cfg.args.fc_float and cfg.args.bias_float:
    def if_binary(n):
        return (not('bn' in n) and not('bias' in n))
else:
    def if_binary(n):
        return (not('bn' in n))


if cfg.args.fc_float and cfg.args.bias_float:
    def if_binary_tern(n):
        return (not('bn' in n) and not('fc' in n) and not('linear' in n) and not('bias' in n))
elif not cfg.args.fc_float and cfg.args.bias_float:
    def if_binary_tern(n):
        return (not('bn' in n) and not('bias' in n))
else:
    def if_binary_tern(n):
        return (not('bn' in n))
