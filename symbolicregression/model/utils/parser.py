import os
import time
import random
import logging
from argparse import ArgumentParser
from .attr_dict import AttrDict
from datetime import datetime

logger = logging.getLogger('my.utils.parser')


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', choices=['default', 'hard', 'pure', 'keep', 'load'], default='default')
    parser.add_argument('-n', '--name', type=str, default='default', help='')
    parser.add_argument('-s', '--seed', type=int, default=42)
    parser.add_argument('--tot_steps', type=int, default=300000)
    parser.add_argument('--device', type=str, default='auto', help='cpu | cuda | cuda:0 | auto | cuda:0,1,2')
    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_input', type=int, default=512)
    parser.add_argument('--d_output', type=int, default=512)
    parser.add_argument('--n_TE_layers', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--max_param', type=int, default=5)
    parser.add_argument('--max_var', type=int, default=10)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--save_equations', type=int, default=0, help='Save N generated equations')
    parser.add_argument('--AMP', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--uniform_sample_number', type=int, default=400, help='If 0, sample 100~500 data points')
    parser.add_argument('--pred_loss_weight', type=float, default=1.0)
    parser.add_argument('--clip_loss_weight', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=0) 

    parser.add_argument('--no-simplify', action='store_false', dest='simplify')
    parser.add_argument('--simplify', action='store_true')

    parser.add_argument('--no-normalize_loss', action='store_false', dest='normalize_loss')
    parser.add_argument('--normalize_loss', action='store_true')

    parser.add_argument('--no-normalize_X', action='store_false', dest='normalize_X')
    parser.add_argument('--normalize_X', action='store_true')
    parser.add_argument('--no-normalize_y', action='store_false', dest='normalize_y')
    parser.add_argument('--normalize_y', action='store_true')

    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--use_raw_numerical', type=bool, default=True)
    parser.add_argument('--model_type', type=str,
                        help='Type of VAE model to use (transformer or convolutional)')
    
    parser.add_argument('--hidden_layer_sizes', nargs='+', type=int, default=[24, 48, 96, 192, 192, 192],
                        help='Hidden layer sizes for convolutional VAE')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for convolutional layers')
    parser.add_argument('--stride', type=int, default=2,
                        help='Stride for convolutional layers')
    parser.add_argument('--padding', type=int, default=1,
                        help='Padding for convolutional layers')
    parser.add_argument('--output_padding', type=int, default=1,
                        help='Output padding for transposed convolutional layers')
    parser.add_argument('--dilation', type=int, default=1,
                        help='Dilation factor for convolutional layers')
    parser.add_argument('--groups', type=int, default=1,
                        help='Number of blocked connections from input channels to output channels')
    parser.add_argument('--conv_bias', type=bool, default=True,
                        help='Whether to add a learnable bias to convolutional layers')
    parser.add_argument('--activation', type=str, choices=['relu', 'leaky_relu', 'gelu', 'selu', 'tanh'], default='relu',
                        help='Activation function to use in convolutional layers')
    parser.add_argument('--use_batch_norm', type=bool, default=True,
                        help='Whether to use batch normalization in convolutional layers')
    parser.add_argument('--conv_dropout', type=float, default=0.1,
                        help='Dropout rate for convolutional layers')
    parser.add_argument('--num_conv_layers', type=int, default=3,
                        help='Number of convolutional layers in encoder and decoder')
    parser.add_argument('--channel_multiplier', type=int, default=2,
                        help='Multiplier for number of channels between convolutional layers')
    
    parser.add_argument('--default_dataset_path', type=str, default='./genedata/saved_datasets/default_dataset.pkl')
    parser.add_argument('--num_equations', type=int, default=50)
    parser.add_argument('--gen_dataset', type=bool, default=False)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--test_flag', type=bool, default=False)

    parser.add_argument('--use_norm', type=bool, default=True)

    parser.add_argument('--wandb_flag', type=int, default=1)
    parser.add_argument('--wd_project', type=str, default='SRvae_test')

    parser.add_argument('--reload_model_e2edec', type=str, default='./weights/model1.pt')

    return parser


def parse_parser(parser: ArgumentParser, save_dir='./results'):
    args, unknown = parser.parse_known_args()

    if unknown: logger.warning(f'Unknown args: {unknown}')
    args.seed = args.seed or random.randint(1, 32768)

    args.run_time = f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]}'

    args.save_dir = os.path.join(save_dir, args.name)

    args = AttrDict(vars(args))
    return args


def get_args(save_dir='./results') -> AttrDict:
    parser = get_parser()
    args = parse_parser(parser, save_dir=save_dir)
    return args

