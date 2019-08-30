import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    # data arguments
    parser.add_argument('--data', default='learning/treelstm/data/lc-quad/',
                        help='path to dataset')
    parser.add_argument('--save', default='learning/treelstm/checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='lc_quad',
                        help='Name to identify experiment')
    # model arguments
    parser.add_argument('--mem_dim', default=150, type=int,
                        help='Size of TreeLSTM cell state')
    parser.add_argument('--freeze_embed', action='store_true',
                        help='Freeze word embeddings')
    # training arguments
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=12, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=2.25e-3, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--emblr', default=1e-2, type=float,
                        metavar='EMLR', help='initial embedding learning rate')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adam',
                        help='optimizer (default: adam)')

    # miscellaneous options
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed (default: 42)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args

config = {
    'general': {
        'http': {
            'timeout': 120
        },
        'dbpedia': {
            'endpoint': 'http://dbpedia.org/sparql',
            'one_hop_bloom_file': './data/blooms/spo1.bloom',
            'two_hop_bloom_file': './data/blooms/spo2.bloom'
        }
    }
}

