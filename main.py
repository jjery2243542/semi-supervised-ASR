from solver import Solver
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-mode', '-m', default='train')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    solver = Solver(config)

    if args.mode == 'train':
        solver.sup_train()
    else:
        solver.test()
