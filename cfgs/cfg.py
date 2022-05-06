import argparse

RANDOM_SEED = 123456

INPUT_CHANNELS = 1
IM_SIZE = 28
INPUT_DIM = 3
HIDDEN_DIM = 2
OUTPUT_DIM = 2
DATASET_SIZE = 10
BATCH_SIZE = 64
SAVE_DIR = './out'
GPU_ID = '0'
QUANT_LEVELS = 2    # 2: {-1, 1}, 3: {-1, 0, 1}
PROJECTION = 'SOFTMAX'  # EUCLIDEAN
DATASET = 'MNIST'       # CIFAR10, CIFAR100, TINYIMAGENET200, IMAGENET1000
ARCHITECTURE = 'MLP'    # LENET300, LENET5, VGG16, RESNET18
ROUNDING = 'ARGMAX'     # ICM
EVAL_SET = 'TEST'   # TRAIN
VAL_SET = 'TRAIN'   # TEST
LOSS_FUNCTION = 'HINGE'   # CROSSENTROPY
METHOD = 'ALL' # SIMPLEX, CONTINUOUS, XNOR, PGDSIMPLEX, BNN, MDA
LOG_INTERVAL = 100
SAVE_NAME = ''          # best model file
DATA_PATH = '../Datasets'
PRETRAINED_MODEL = ''   # pre-trained model file
EVAL = ''       # trained model to evaluate
RESUME = ''         # checkpoint file to resume


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Script for binary nets.")
    ###### data settings
    parser.add_argument("--input-channels", type=int, default=INPUT_CHANNELS, help="Input channels.")
    parser.add_argument("--im-size", type=int, default=IM_SIZE, help="Image size.")
    parser.add_argument("--input-dim", type=int, default=INPUT_DIM, help="Input dimension.")
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM, help="Hidden layer size.")
    parser.add_argument("--output-dim", type=int, default=OUTPUT_DIM, help="Output dimension.")
    parser.add_argument("--dataset-size", type=int, default=DATASET_SIZE, help="No of datapoints.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Type of architecture [MNIST, CIFAR10, CIFAR100].")
    parser.add_argument("--architecture", type=str, default=ARCHITECTURE,
                        help="Type of architecture [MLP, LENET300, LENET5, VGG16, RESNET18].")

    parser.add_argument("--save-dir", type=str, default=SAVE_DIR, help="Output directory.")
    parser.add_argument("--gpu-id", type=str, default=GPU_ID, help="Cuda visible device.")
    parser.add_argument("--loss-function", type=str, default=LOSS_FUNCTION,
                        help="Loss function [HINGE, CROSSENTROPY].")

    ###### method settings
    parser.add_argument("--method", type=str, default=METHOD,
                        help="Method to run [ALL, CONTINUOUS, BNN].")
    parser.add_argument("--zeroone", action="store_true", help="Flag to set Q_l = {0,1} in case of binary")
    parser.add_argument("--tanh", action="store_true", help="Flag to perform tanh projection in w space in BNN")
    parser.add_argument("--full-ste", action="store_true", help="Always for full-STE (no call to update_auxgradient())")
    parser.add_argument("--quant-levels", type=int, default=QUANT_LEVELS,
                        help="Quantization levels: {2: {-1, 1}, 3: {-1, 0, 1}}.")
    parser.add_argument("--projection", type=str, default=PROJECTION,
                        help="Type of projection [SOFTMAX, EUCLIDEAN, ARGMAX].")

    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL,
                        help="No of iterations before printing loss.")

    ###### experiment settings
    parser.add_argument("--save-name", type=str, default=SAVE_NAME, help="Name to save the best model.")
    parser.add_argument("--pretrained-model", type=str, default=PRETRAINED_MODEL, help="Pretrained model to load weights.")
    parser.add_argument("--data-path", type=str, default=DATA_PATH, help="Path to store datasets.")
    parser.add_argument("--eval", type=str, default=EVAL, help="Model file to evaluate.")
    parser.add_argument("--resume", action="store_true", help="to resume training.")
    parser.add_argument("--resume_file", type=str, default=RESUME, help="Checkpoint file to resume training.")
    parser.add_argument("--eval-set", type=str, default=EVAL_SET, help="Dataset to evaluate [TEST, TRAIN].")
    parser.add_argument("--val-set", type=str, default=VAL_SET, help="Dataset to validate [TEST, TRAIN].")
    parser.add_argument("--use_tensorboard", action="store_true", help="Flag to Plot histogram of weights over a period "
                                                                       "of epochs using tensorboard")
    parser.add_argument("--exp_name", type=str, default='', help="experiment name.")
    parser.add_argument("--bn_affine", action="store_true", help="batch norm affine")
    parser.add_argument("--bias_float", action="store_true", help="learn bias in float")
    parser.add_argument("--fc_float", action="store_true", help="learn FC layer in floating")

    ###### adversarial attack parameters
    parser.add_argument("--eval_adv", type=str, default='', help="Adv. Model file to evaluate.")
    parser.add_argument("--attack_iters", type=int, default=20, help="number of iterations used for iterative PGD attack")
    parser.add_argument("--attack_radius", type=float, default=8.0, help="radius for the bound on the adversarial example")
    parser.add_argument("--attack_stepsize", type=float, default=2.0, help="learning rate for PGD attack")

    ##### improve signal propagation
    parser.add_argument("--sp_fixed_scalar", action="store_true", help="improve signal propagation by multiplying with"
                                                                       "a fixed scalar")
    parser.add_argument("--modified_pgd_attack", action="store_true", help="modified PGD by using mean singular value")
    parser.add_argument('--attack', default=None, type=str, choices=['l2_pgd', 'linf_pgd', 'fgsm'])
    parser.add_argument('--modified_attack_technique', default=None, type=str, choices=['TS_JSV_network', 'TS_grad_thresh', 'TS_gradthresh_MJSVnetwork',
                                                                                        'Beta_NonLinearity_hessorig'])
    parser.add_argument("--gradthresh_epsilon", type=float, default=0.01, help="gradient threshold for modified PGD attack")
    parser.add_argument("--random-restarts", type=float, default=1, help="Number of random restarts")

    return parser.parse_args()


args = get_arguments()