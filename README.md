# Improved Gradient-Based Adversarial Attacks for Quantized Networks

This repository is the official implementation of AAAI 2022 paper: [Improved Gradient-Based Adversarial Attacks for Quantized Networks](https://arxiv.org/pdf/2003.13511.pdf).

This code is for research purpose only.

Any questions or discussions are welcomed!

## Installation and Setup

Setup python virtual environment.

```
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the pre-trained FP32, BNN-WQ and BNN-WAQ models on different datasets and architectures from [here](https://drive.google.com/file/d/1idPhNMgQ4SZ_8EHv0zciQoe-t2eqXxCl/view?usp=sharing).

## Adversarial accuracy evaluation of FGSM and FGSM++ (NJS/HNS) on FP32, BNN-WQ and BNN-WAQ

Shell scripts to evaluate FGSM/FGSM++ adversarial accuracy of pre-trained networks (obtained from [1]) can be found in `shell_scripts/fgsm` folder. 

```
sh shell_scripts/fgsm/cifar10.sh
sh shell_scripts/fgsm/cifar100.sh
```

## Adversarial accuracy evaluation of PGD L2 and PGD++ L2 (NJS/HNS) on FP32, BNN-WQ and BNN-WAQ

Shell scripts to evaluate PGD L2 / PGD++ L2 adversarial accuracy of pre-trained networks (obtained from [1]) can be found in `shell_scripts/pgd-l2` folder. 

```
sh shell_scripts/pgd-l2/cifar10.sh
sh shell_scripts/pgd-l2/cifar100.sh
```

## Adversarial accuracy evaluation of PGD LInf and PGD++ LInf (NJS/HNS) on FP32, BNN-WQ and BNN-WAQ

Shell scripts to evaluate PGD LInf / PGD++ LInf adversarial accuracy of pre-trained networks (obtained from [1]) can be found in `shell_scripts/pgd-linf` folder. 

```
sh shell_scripts/pgd-linf/cifar10.sh
sh shell_scripts/pgd-linf/cifar100.sh
```

NOTE: The results may vary slightly based on which cuda, torch, torchvision versions you use.

## Cite

If you make use of this code in your own work, please cite our papers:

```
@article{gupta2020improved,
  title={Improved gradient based adversarial attacks for quantized networks},
  author={Gupta, Kartik and Ajanthan, Thalaiyasingam},
  journal={arXiv preprint arXiv:2003.13511},
  year={2020}
}
@inproceedings{ajanthan2019mirror,
  title={Mirror descent view for neural network quantization},
  author={Ajanthan, Thalaiyasingam and Gupta, Kartik and Torr, Philip HS and Hartley, Richard and Dokania, Puneet K},
  booktitle={Artificial intelligence and statistics},
  year={2021},
  organization={PMLR}
}
```

#### Contact
Kartik Gupta (kartik.gupta@anu.edu.au).


References
----------------------
[1] Ajanthan, Thalaiyasingam and Gupta, Kartik and Torr, Philip HS and Hartley, Richard and Dokania, Puneet K. Mirror descent view for neural network quantization. AISTATS 2021.
