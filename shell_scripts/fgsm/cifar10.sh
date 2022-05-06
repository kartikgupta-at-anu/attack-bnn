#!/usr/bin/env bash

sd=./out
ep=0    #0:use iters
bs=128
lf='CROSSENTROPY'
mm=0.9
lri=30000
iters=100000
wd=0.0001
pr='SOFTMAX'
op='SGD'
lr=0.1
lrs=0.1
bti=100
bts=1.00


attack='fgsm'

gpu_id=9

####### REF (FP32) #############

dt='CIFAR10'
ar='RESNET18'
mt='CONTINUOUS'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --eval_adv out/$dt/$ar/$mt/best_model_adv.pth.tar --attack $attack

technique='TS_gradthresh_MJSVnetwork'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --eval_adv out/$dt/$ar/$mt/best_model_adv.pth.tar --attack $attack --modified_attack_technique $technique

technique='Beta_NonLinearity_hessorig'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --eval_adv out/$dt/$ar/$mt/best_model_adv.pth.tar --attack $attack --modified_attack_technique $technique


dt='CIFAR10'
ar='VGG16'
mt='CONTINUOUS'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack

technique='TS_gradthresh_MJSVnetwork'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique

technique='Beta_NonLinearity_hessorig'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique



####### BNN-WQ #############

dt='CIFAR10'
ar='RESNET18'
mt='BNN'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --eval out/$dt/$ar/$mt/best_model.pth.tar --eval_adv out/$dt/$ar/$mt/best_model_adv.pth.tar --attack $attack

technique='TS_gradthresh_MJSVnetwork'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --eval out/$dt/$ar/$mt/best_model.pth.tar --eval_adv out/$dt/$ar/$mt/best_model_adv.pth.tar --attack $attack --modified_attack_technique $technique

technique='Beta_NonLinearity_hessorig'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --eval out/$dt/$ar/$mt/best_model.pth.tar --eval_adv out/$dt/$ar/$mt/best_model_adv.pth.tar --attack $attack --modified_attack_technique $technique


dt='CIFAR10'
ar='VGG16'
mt='BNN'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack

technique='TS_gradthresh_MJSVnetwork'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique

technique='Beta_NonLinearity_hessorig'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --tanh --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique



####### BNN-WAQ #############

dt='CIFAR10'
ar='resnet_binary'
mt='BNN_WAQ'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack

technique='TS_gradthresh_MJSVnetwork'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique

technique='Beta_NonLinearity_hessorig'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique


dt='CIFAR10'
ar='vgg_cifar10_binary'
mt='BNN_WAQ'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack

technique='TS_gradthresh_MJSVnetwork'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique

technique='Beta_NonLinearity_hessorig'

python iga_eval.py --save-dir $sd --gpu-id $gpu_id --architecture $ar --loss-function $lf --method $mt --projection $pr --dataset $dt --batch-size $bs --eval out/$dt/$ar/$mt/best_model.pth.tar --attack $attack --modified_attack_technique $technique

