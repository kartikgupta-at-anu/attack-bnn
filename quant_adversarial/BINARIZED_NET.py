""" binarized net implementation -- NIPS paper!
"""

import logging
import os
import models
import utils.utils as util
import torch
from quant_adversarial.linf_pgd_attack import pgd, modified_pgd
from quant_adversarial.METHODS import _METHODS


BETAMAX = 10000


class _Binarized_Nets(_METHODS):
    def __init__(self, args):
        # super(_Continuous_Nets).__init__(args)
        _METHODS.__init__(self, args)
        print('\n#### Running binarized-net ####')
        self.args = args

    def evaluate(self, amodel, model, device, loader, best_acc, training=False, beta=1., save_name=None):
        model.eval()
        correct1 = 0
        correct5 = 0
        tsize = 0

        if training:
            # store aux-weights
            amodel.store(model)
            # binarize
            #        if args.tanh:
            #            dotanh(args, model, beta=beta)  # no need as tanh does not change sign!
            self.doround(model)

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
            # restore aux-weights
            amodel.restore(model)
            model.train()

        acc1 = 100. * correct1 / tsize
        acc5 = 100. * correct5 / tsize
        if (acc1 > best_acc):
            best_acc = acc1.item()
            if training:    # storing the continuous weights of the best model, done separately from checkpoint!
                util.save_model({'state_dict': model.state_dict(), 'best_acc1': best_acc}, save_name)

        return acc1.item(), acc5.item(), best_acc

    def evaluate_val_adv(self, amodel, model, device, loader, noise_radius, attack_step_size, pytorch_range,
                     best_acc, training=False, beta=1., save_name=None):
        """ evaluate the model given data
        """
        model.eval()
        correct1 = 0
        correct5 = 0
        tsize = 0

        if training:
            amodel.store(model)
            self.doround(model)

        for data, target in loader:
            data, target = data.to(device, torch.float), target.to(device, torch.long)
            data_pert = data
            pgd_iter = 7

            if self.args.modified_pgd_attack:
                r = modified_pgd(data, model.eval(), self.args, sp_scalar=1.0, epsilon=[noise_radius], targets=target, pytorch_range=pytorch_range,
                                 step_size=attack_step_size,
                                 num_steps=pgd_iter, epsil=noise_radius)
            else:
                r = pgd(data, model.eval(), epsilon=[noise_radius], targets=target, pytorch_range=pytorch_range,
                        step_size=attack_step_size,
                        num_steps=pgd_iter, epsil=noise_radius)
            data_pert = data_pert + r
            output = model(data_pert)
            # topk accuracy
            c1, c5 = util.accuracy(output.data, target, topk=(1, 5))
            correct1 += c1
            correct5 += c5
            tsize += target.size(0)

        if training:
            # restore aux-weights
            amodel.restore(model)
            model.train()

        acc1 = 100. * correct1 / tsize
        acc5 = 100. * correct5 / tsize
        if (acc1 > best_acc):
            best_acc = acc1.item()
            if training:    # storing the continuous weights of the best model, done separately from checkpoint!
                util.save_model({'state_dict': model.state_dict(), 'best_acc1': best_acc, 'beta': beta}, save_name)

        return acc1.item(), acc5.item(), best_acc

    def doround(self, model):
        """ binarize
        """
        for i, (name, p) in enumerate(model.named_parameters()):
            if self.args.quant_levels == 2:
                if util.if_binary(name):
                    p.data = p.data.sign()  # sign() of 0 is 0
            elif self.args.quant_levels == 3:    # ternary
                p.data[p.data.le(-0.5)] = -1
                p.data[p.data.gt(-0.5) * p.data.lt(0.5)] = 0
                p.data[p.data.ge(0.5)] = 1
            else:
                assert(0)

    def dotanh(self, model, beta=1.):
        """ tanh projection
        """
        for i, (name, p) in enumerate(model.named_parameters()):
            if self.args.quant_levels == 2:  # binary
                if util.if_binary(name):
                    p.data = torch.tanh(p.data * beta)
            elif self.args.quant_levels == 3:    # ternary, shifted tanh
                p.data = 0.5 * (torch.tanh(beta*(p.data + 0.5)) + torch.tanh(beta*(p.data - 0.5)))
            else:
                assert(0)
