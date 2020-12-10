import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist | cifar10 | cifar100',
                        choices=('mnist', 'cifar10', 'cifar100'))
    parser.add_argument('--flatten', type=bool, default=False, help='flatten the input data')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--feat_dims', type=int, default=128, help='dimensions of the latent features')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
    parser.add_argument('--verbose', type=bool, default=True, help='print training process')
    args = parser.parse_args(argv)
    return args