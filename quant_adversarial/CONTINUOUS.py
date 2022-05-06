""" continuous classifier 
"""

import logging
import os
import models
import utils.utils as util
import torch
from quant_adversarial.linf_pgd_attack import pgd, modified_pgd
from quant_adversarial.METHODS import _METHODS


class _Continuous_Nets(_METHODS):
    def __init__(self, args):
        # super(_Continuous_Nets).__init__(args)
        _METHODS.__init__(self, args)
        print('\n#### Running continuous-net ####')
        self.args = args

    def evaluate(self, model, device, loader, best_acc, save_name, training=False):
        model.eval()
        correct1 = 0
        correct5 = 0
        tsize = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device, torch.float), target.to(device, torch.long)
                output = model(data)
                # topk accuracy
                c1, c5 = util.accuracy(output.data, target, topk=(1, 5))
                correct1 += c1
                correct5 += c5
                tsize += target.size(0)

        if training:
            model.train()

        acc1 = 100. * correct1 / tsize
        acc5 = 100. * correct5 / tsize
        if (acc1 > best_acc):
            best_acc = acc1.item()
            if training:    # storing the continuous weights of the best model, done separately from checkpoint!
                util.save_model({'state_dict': model.state_dict(), 'best_acc1': best_acc}, save_name)

        return acc1.item(), acc5.item(), best_acc

    def evaluate_val_adv(self, model, device, loader, noise_radius, attack_step_size, pytorch_range, best_acc,
                         save_name, training=False):
        model.eval()
        correct1 = 0
        correct5 = 0
        tsize = 0
        # with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device, torch.float), target.to(device, torch.long)

            data_pert = data
            pgd_iter = 7

            if self.args.modified_pgd_attack:
                r = modified_pgd(data, model.eval(), self.args, sp_scalar=1.0, epsilon=[noise_radius], targets=target, pytorch_range=pytorch_range,
                                 step_size=attack_step_size,
                                 num_steps=pgd_iter, epsil=noise_radius)
            else:
                r = pgd(data, model.eval(), epsilon=[noise_radius], targets=target, pytorch_range=pytorch_range, step_size=attack_step_size,
                        num_steps=pgd_iter, epsil=noise_radius)
            data_pert = data_pert + r

            output = model(data_pert)
            # topk accuracy
            c1, c5 = util.accuracy(output.data, target, topk=(1, 5))
            correct1 += c1
            correct5 += c5
            tsize += target.size(0)

        if training:
            model.train()

        acc1 = 100. * correct1 / tsize
        acc5 = 100. * correct5 / tsize
        if (acc1 > best_acc):
            best_acc = acc1.item()
            if training:    # storing the continuous weights of the best model, done separately from checkpoint!
                util.save_model({'state_dict': model.state_dict(), 'best_acc1': best_acc}, save_name)

        return acc1.item(), acc5.item(), best_acc

