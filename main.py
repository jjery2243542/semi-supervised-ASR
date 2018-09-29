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

    if args.sup_train or args.judge_train:
        solver = Solver(config, mode='train')
    else:
        solver = Solver(config)

    if args.load_model:
        solver.load_model(config['load_model_path'], config['load_optimizer'])
    if args.load_judge:
        solver.load_judge(config['load_model_path'], config['load_optimizer'])

    if args.sup_train:
        state_dict, cer = solver.sup_pretrain()

    if args.judge_train:
        solver.judge_pretrain()

    if args.test and args.sup_train:
        solver.test(state_dict=state_dict)
    elif args.test:
        solver.test()
