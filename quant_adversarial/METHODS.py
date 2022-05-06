import models, logging, os
import utils.utils as util
import torch, yaml,time, errno
import torch.nn as nn
import torch.optim as optim
import numpy as np
import fcntl
import datetime, time

from utils.visualise_adv import jacobian_faster
from quant_adversarial.linf_pgd_attack import pgd, modified_pgd
from quant_adversarial.l2_pgd_attack import pgd_l2, modified_pgd_l2
from quant_adversarial.fgsm_attack import fgsm, modified_fgsm

class _METHODS():
    def __init__(self, args):
        self.args = args

    def model_def(self, device):
        # architecture
        if 'VGG' in self.args.architecture:
            assert(self.args.architecture == 'VGG11' or self.args.architecture == 'VGG13' or self.args.architecture == 'VGG16'
                   or self.args.architecture == 'VGG19')
            model = models.VGG(self.args.architecture, self.args.input_channels, self.args.im_size, self.args.output_dim).to(device)
            model_adv = models.VGG(self.args.architecture, self.args.input_channels, self.args.im_size, self.args.output_dim).to(device)
        elif self.args.architecture == 'RESNET18':
            model = models.ResNet18(self.args.input_channels, self.args.im_size, self.args.output_dim).to(device)
            model_adv = models.ResNet18(self.args.input_channels, self.args.im_size, self.args.output_dim).to(device)
        else:
            print('Architecture type "{0}" not recognized, exiting ...'.format(self.args.architecture))
            exit()
        return model, model_adv

    def loss_def(self):
        if self.args.loss_function == 'HINGE':
            criterion = nn.MultiLabelMarginLoss()
        elif self.args.loss_function == 'CROSSENTROPY':
            criterion = nn.CrossEntropyLoss()
        else:
            print('Loss type "{0}" not recognized, exiting ...'.format(self.args.loss_function))
            exit()
        return criterion

    def load_model(self, model):
        logging.info('Loading checkpoint file "{0}" for evaluation'.format(self.args.eval))
        if not os.path.isfile(self.args.eval):
            print('Checkpoint file "{0}" for evaluation not recognized, exiting ...'.format(self.args.eval))
            exit()
        checkpoint = torch.load(self.args.eval)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def load_model_adv(self, model):
        if self.args.eval_adv is not '':
            logging.info('Loading checkpoint file "{0}" for evaluation'.format(self.args.eval_adv))
            if not os.path.isfile(self.args.eval_adv):
                print('Checkpoint file "{0}" for evaluation not recognized, exiting ...'.format(self.args.eval_adv))
                exit()
            checkpoint = torch.load(self.args.eval_adv)
        else:
            logging.info('Loading checkpoint file "{0}" for evaluation'.format(self.args.eval[:-8]+'_adv.pth.tar'))
            checkpoint = torch.load(self.args.eval[:-8]+'_adv.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def evaluate_adv(self, model, device, test_loader, test_loader_jac, noise_radius, attack_step_size, pytorch_range,
                     adv_training, sp_fixed_scalar=False, random_restarts=1):
        model.eval()

        test_loss, adv_acc, total, curvature, clean_acc, grad_sum = 0, 0, 0, 0, 0, 0

        start_time = time.time()

        if self.args.modified_attack_technique=='TS_JSV_network' or self.args.modified_attack_technique=='TS_gradthresh_MJSVnetwork':
            sp_scalar = self.jacobian_io(test_loader_jac, device, model, adv_training)
            print("MJSV: ", sp_scalar)
        else:
            sp_scalar = 1.0

        logits_list = []
        labels_list = []

        d1 = torch.full((len(test_loader.dataset),), np.inf)
        d2 = torch.full((len(test_loader.dataset),), np.inf)
        dinf = torch.full((len(test_loader.dataset),), np.inf)

        d1 = d1.cuda()
        d2 = d2.cuda()
        dinf = dinf.cuda()

        K = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # print("Batch id for attack: ", batch_idx)
            Nb = len(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model.eval()(inputs)

            logits_list.append(outputs.detach())
            labels_list.append(targets.detach())

            _, predicted = outputs.max(1)
            clean_acc += predicted.eq(targets).sum().item()
            total += targets.size(0)

            for restart in range(int(random_restarts)):
                #### PGD Attack
                inputs_pert = inputs + 0.
                # noise_radius = 5./255.*8
                pgd_iter = self.args.attack_iters

                if self.args.modified_pgd_attack:
                    r = modified_pgd(inputs, model.eval(), self.args, sp_scalar, epsilon=[noise_radius], targets=targets, pytorch_range=pytorch_range,
                            step_size=attack_step_size,
                            num_steps=pgd_iter, epsil=noise_radius)
                elif self.args.attack is None:
                    r = pgd(inputs, model.eval(), epsilon=[noise_radius], targets=targets, pytorch_range=pytorch_range,
                            step_size=attack_step_size,
                            num_steps=pgd_iter, epsil=noise_radius)

                if self.args.attack == 'linf_pgd':
                    if self.args.modified_attack_technique is None:
                        r = pgd(inputs, model.eval(), epsilon=[noise_radius], targets=targets, pytorch_range=pytorch_range,
                                step_size=attack_step_size,
                                num_steps=pgd_iter, epsil=noise_radius)
                    else:
                        r = modified_pgd(inputs, model.eval(), self.args, sp_scalar, epsilon=[noise_radius], targets=targets, pytorch_range=pytorch_range,
                                         step_size=attack_step_size,
                                         num_steps=pgd_iter, epsil=noise_radius)
                elif self.args.attack == 'l2_pgd':
                    if self.args.modified_attack_technique is None:
                        r = pgd_l2(inputs, model.eval(), epsilon=[noise_radius], targets=targets, pytorch_range=pytorch_range,
                                step_size=0.0,
                                num_steps=pgd_iter, epsil=noise_radius)
                    else:
                        r = modified_pgd_l2(inputs, model.eval(), self.args, sp_scalar, epsilon=[noise_radius], targets=targets, pytorch_range=pytorch_range,
                                         step_size=0.0,
                                         num_steps=pgd_iter, epsil=noise_radius)
                elif self.args.attack == 'fgsm':
                    if self.args.modified_attack_technique is None:
                        r = fgsm(inputs, model.eval(), targets=targets, pytorch_range=pytorch_range,
                                   step_size=noise_radius,
                                   epsil=noise_radius)
                    else:
                        r = modified_fgsm(inputs, model.eval(), self.args, sp_scalar, targets=targets, pytorch_range=pytorch_range,
                                            step_size=noise_radius,
                                            epsil=noise_radius)

                r = r.detach()
                with torch.no_grad():
                    inputs_pert = inputs_pert + r
                    outputs = model(inputs_pert)
                probs, predicted = outputs.max(1)
                if not predicted.eq(targets).sum().item():
                    break;
            probs, predicted = outputs.max(1)
            adv_acc += predicted.eq(targets).sum().item()
            l1 = r.view(Nb, -1).norm(p=1, dim=-1)
            l2 = r.view(Nb, -1).norm(p=2, dim=-1)
            linf = r.view(Nb, -1).norm(p=np.inf, dim=-1)
            ix = torch.arange(K, K+Nb, device=inputs.device)

            d1[ix] = l1
            d2[ix] = l2
            dinf[ix] = linf
            K += Nb

        executionTime = (time.time() - start_time)
        print('Adversarial accuracy Execution time in seconds: {}'.format(executionTime))


        logging.info('Attack Config--> '
                     'Iterations: {iter:.2f}, Radius: {radius:.2f}, Step Size: '
                     '{step_size:.2f}, Epsilon: {epsilon:.6f}'.format(
            iter=self.args.attack_iters, radius=self.args.attack_radius,
            step_size=self.args.attack_stepsize, epsilon=self.args.gradthresh_epsilon))

        if adv_training:
            if sp_fixed_scalar:
                logging.info('w/ Adversarial Training w/ SP fixed scalar--> '
                             'Accuracy: Clean: {clean_acc:.2f}, PGD Adversarial: {adv_acc:.2f}%'.format(
                    clean_acc=100.*clean_acc/total, adv_acc=100.*adv_acc/total))
            else:
                logging.info('w/ Adversarial Training w/o SP fixed scalar--> '
                             'Accuracy: Clean: {clean_acc:.2f}, PGD Adversarial: {adv_acc:.2f}%'.format(
                    clean_acc=100.*clean_acc/total, adv_acc=100.*adv_acc/total))
        else:
            if sp_fixed_scalar:
                logging.info('w/o Adversarial Training w/ SP fixed scalar--> '
                             'Accuracy: Clean: {clean_acc:.2f}, PGD Adversarial: {adv_acc:.2f}%'.format(
                    clean_acc=100.*clean_acc/total, adv_acc=100.*adv_acc/total))
            else:
                logging.info('w/o Adversarial Training w/o SP fixed scalar--> '
                             'Accuracy: Clean: {clean_acc:.2f}, PGD Adversarial: {adv_acc:.2f}%'.format(
                    clean_acc=100.*clean_acc/total, adv_acc=100.*adv_acc/total))

        # # # ####################### adv bnn eval storage extensive
        save_dir = os.path.join('attack_out', self.args.dataset, self.args.architecture, self.args.attack)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.args.modified_attack_technique is not None:
            result_filename = 'Modified_' + self.args.modified_attack_technique
        else:
            result_filename = 'Original'

        result_filename = os.path.join(save_dir, result_filename + '.yml')

        if adv_training:
            if self.args.full_ste:
                method_name = self.args.method+'_STE_adv'
            else:
                method_name = self.args.method+'_adv'
        else:
            if self.args.full_ste:
                method_name = self.args.method+'_STE'
            else:
                method_name = self.args.method

        if os.path.exists(result_filename):
            outfile = open(result_filename, 'r+')
            data = yaml.safe_load(outfile)
        else:
            data = {}
        if os.path.exists(result_filename):
            outfile = open(result_filename, 'r+')

            while True:
                try:
                    fcntl.flock(outfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError as e:
                    # raise on unrelated IOErrors
                    if e.errno != errno.EAGAIN:
                        raise
                    else:
                        print("Waiting for Result file Lock to be Free !!!")
                        time.sleep(0.5)
            data = yaml.safe_load(outfile)
            lock_notneeded = False
        else:
            lock_notneeded = True
            data = {}

        data[method_name] = {}
        data[method_name]['clean'] = round(float(100.*clean_acc/total), 4)
        data[method_name]['adv'] = round(float(100.*adv_acc/total), 4)

        with open(result_filename, "w") as outfile_write:
            yaml.dump(data, outfile_write, default_flow_style=False)

        if not lock_notneeded:
            fcntl.flock(outfile, fcntl.LOCK_UN)
        # #######################

        mn2 = d2.mean()
        md2 = d2.median()
        mx2 = d2.max()
        mninf = dinf.mean()
        mdinf = dinf.median()
        mxinf = dinf.max()
        logging.info('Statistics in L2 norm: Mean: {mean:.4f}, Median: {median:.4f}, Max: {max:.4f}'.format(
            mean=mn2, median=md2, max=mx2))
        logging.info('Statistics in Linf norm: Mean: {mean:.4f}, Median: {median:.4f}, Max: {max:.4f}'.format(
            mean=mninf, median=mdinf, max=mxinf))

    def jacobian_io(self, test_loader, device, model, adv_training=False, sp_fixed_scalar=False):
        model.eval()
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # inputs.requires_grad_()
            ### just to find the number of classes
            if i == 0:
                with torch.no_grad():
                    outputs = model.eval()(inputs)
            # begin_time = datetime.datetime.now()
            # jacobian_mat1 = jacobian(outputs, inputs)
            # print("Time for loop jacobian:", datetime.datetime.now() - begin_time)
            # begin_time = datetime.datetime.now()
            jacobian = jacobian_faster(model, inputs, outputs.shape[1])
            # print("Time for faster jacobian:", datetime.datetime.now() - begin_time)
            u, s, v = torch.svd(jacobian)

            if i==0:
                singular_val = s
                avg_s = s.mean().unsqueeze(0)
            else:
                singular_val = torch.cat((singular_val, s))
                avg_s = torch.cat((avg_s, s.mean().unsqueeze(0)))

            ##### break after 100 samples
            if i==100:
                break;

        sv_avg = singular_val.cpu().data.numpy()

        logging.info('### JACOBIAN Singular Values ###')
        if adv_training:
            if sp_fixed_scalar:
                logging.info('w/ Adversarial Training and w/ SP fixed scalar --> '
                             'Mean: {mean_sv:.4f}  Standard Deviation: {std_sv:.4f} Condition Number: {cn_sv:.4f}'
                             .format(mean_sv=np.mean(sv_avg), std_sv=np.std(sv_avg), cn_sv=np.amax(sv_avg)/np.amin(sv_avg)))
                logging.info(sv_avg)
            else:
                logging.info('w/ Adversarial Training and w/o SP fixed scalar --> '
                             'Mean: {mean_sv:.4f}  Standard Deviation: {std_sv:.4f} Condition Number: {cn_sv:.4f}'
                             .format(mean_sv=np.mean(sv_avg), std_sv=np.std(sv_avg), cn_sv=np.amax(sv_avg)/np.amin(sv_avg)))
                logging.info(sv_avg)
                # np.save(os.path.join(self.args.save_dir, 'jacobian_sv_avg_ADV'), sv_avg)
        else:
            if sp_fixed_scalar:
                logging.info('w/o Adversarial Training and w/ SP fixed scalar --> '
                             'Mean: {mean_sv:.4f}  Standard Deviation: {std_sv:.4f} Condition Number: {cn_sv:.4f}'
                             .format(mean_sv=np.mean(sv_avg), std_sv=np.std(sv_avg), cn_sv=np.amax(sv_avg)/np.amin(sv_avg)))
                logging.info(sv_avg)
            else:
                logging.info('w/o Adversarial Training and w/o SP fixed scalar --> '
                             'Mean: {mean_sv:.4f}  Standard Deviation: {std_sv:.4f} Condition Number: {cn_sv:.4f}'
                             .format(mean_sv=np.mean(sv_avg), std_sv=np.std(sv_avg), cn_sv=np.amax(sv_avg)/np.amin(sv_avg)))
                logging.info(sv_avg)
                # np.save(os.path.join(self.args.save_dir, 'jacobian_sv_avg'), sv_avg)
        return np.mean(sv_avg)
