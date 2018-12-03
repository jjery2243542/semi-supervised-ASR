from solver import Solver
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('--sup_pretrain', action='store_true')
    parser.add_argument('--judge_pretrain', action='store_true')
    parser.add_argument('--ssl_train', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_judge', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    if args.load_model:
        solver = Solver(config, load_model=True)
    else:
        solver = Solver(config, load_model=False)

    if args.load_judge:
        solver.load_judge(config['load_judge_path'], config['load_optimizer'])

    if args.sup_pretrain:
        state_dict, cer = solver.sup_pretrain()

    if args.judge_pretrain:
        solver.judge_pretrain()

    if args.ssl_train:
        solver.ssl_train()

    if args.test:
        solver.test()
