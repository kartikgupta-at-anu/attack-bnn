""" binarized net implementation -- NIPS paper!
"""

import logging
import os
import models
import utils.utils as util
import torch
from quant_adversarial.METHODS import _METHODS
from quant_adversarial.linf_pgd_attack import pgd, modified_pgd


BETAMAX = 10000


class _BNN_WAQ_Nets(_METHODS):
    def __init__(self, args):
        # super(_Continuous_Nets).__init__(args)
        _METHODS.__init__(self, args)
        print('\n#### Running BNN with both weights and activations quantized ####')
        self.args = args

    def model_def(self, device):
        # architecture
        logging.info("creating model %s", self.args.architecture)
        model = models.__dict__[self.args.architecture]
        model_config = {'input_size': self.args.im_size, 'dataset': self.args.dataset.lower()}

        # if args.model_config is not '':
        #     model_config = dict(model_config, **literal_eval(args.model_config))

        model1 = model(**model_config).to(device)
        model2 = model(**model_config).to(device)
        return model1, model2

    def forward(self, data_loader, model, device, criterion, best_acc, epoch=0, training=True, optimizer=None, adv_training=False):
        correct1 = 0
        correct5 = 0
        total_loss = 0
        tsize = 0

        for i, (inputs, target) in enumerate(data_loader):
            if not training:
                with torch.no_grad():
                    input_var, target_var = inputs.to(device, torch.float), target.to(device, torch.long)

                if adv_training:
                    if self.args.modified_pgd_attack:
                        r = modified_pgd(input_var, model.eval(), self.args, sp_scalar=1.0, epsilon=[self.noise_radius], targets=target_var, pytorch_range=self.pytorch_range,
                                         step_size=self.attack_step_size,
                                         num_steps=7, epsil=self.noise_radius)
                    else:
                        r = pgd(input_var, model.eval(), epsilon=[self.noise_radius], targets=target_var, pytorch_range=self.pytorch_range,
                                step_size=self.attack_step_size,
                                num_steps=7, epsil=self.noise_radius)
                    input_var = input_var + r
                    model.eval()

                with torch.no_grad():
                    # compute output
                    output = model(input_var)
            else:
                input_var, target_var = inputs.to(device, torch.float), target.to(device, torch.long)
                if adv_training:
                    if self.args.modified_pgd_attack:
                        r = modified_pgd(input_var, model.eval(), self.args, sp_scalar=1.0, epsilon=[self.noise_radius], targets=target_var, pytorch_range=self.pytorch_range,
                                         step_size=self.attack_step_size,
                                         num_steps=7, epsil=self.noise_radius)
                    else:
                        r = pgd(input_var, model.eval(), epsilon=[self.noise_radius], targets=target_var, pytorch_range=self.pytorch_range,
                                step_size=self.attack_step_size,
                                num_steps=7, epsil=self.noise_radius)
                    input_var = input_var + r
                    model.train()

                input_var.requires_grad_(True)
                # compute output
                output = model(input_var)

            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = util.accuracy(output.data, target_var, topk=(1, 5))

            # topk accuracy
            c1, c5 = util.accuracy(output.data, target_var, topk=(1, 5))
            correct1 += c1
            correct5 += c5
            total_loss += loss.item()
            tsize += target.size(0)

            # losses.update(loss.item(), inputs.size(0))
            # top1.update(prec1.item(), inputs.size(0))
            # top5.update(prec5.item(), inputs.size(0))

            if training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                for p in list(model.parameters()):
                    if hasattr(p,'org'):
                        p.data.copy_(p.org)
                optimizer.step()
                for p in list(model.parameters()):
                    if hasattr(p,'org'):
                        p.org.copy_(p.data.clamp_(-1,1))

        acc1 = 100. * correct1 / tsize
        acc5 = 100. * correct5 / tsize
        total_loss = total_loss / len(data_loader)

        if not training:
            if adv_training:
                if (acc1 > best_acc):
                    best_acc = acc1.item()
                    util.save_model({'state_dict': model.state_dict(), 'best_acc1': best_acc, 'beta': 1}, self.args.save_name_adv)
            else:
                if (acc1 > best_acc):
                    best_acc = acc1.item()
                    util.save_model({'state_dict': model.state_dict(), 'best_acc1': best_acc, 'beta': 1}, self.args.save_name)

        return total_loss, acc1.item(), acc5.item(), best_acc

    def evaluate_epoch(self, data_loader, model, device, criterion, best_acc, epoch, adv_training=False):
        # switch to evaluate mode
        model.eval()
        return self.forward(data_loader, model, device, criterion, best_acc, epoch,
                       training=False, optimizer=None, adv_training=adv_training)

    def doround(self, model):
        """ binarize
        """
        for i, p in enumerate(model.parameters()):
            if self.args.quant_levels == 2:
                p.data = p.data.sign()  # sign() of 0 is 0
            elif self.args.quant_levels == 3:    # ternary
                p.data[p.data.le(-0.5)] = -1
                p.data[p.data.gt(-0.5) * p.data.lt(0.5)] = 0
                p.data[p.data.ge(0.5)] = 1
            else:
                assert(0)

