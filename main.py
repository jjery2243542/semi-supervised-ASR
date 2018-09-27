from solver import Solver
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)

    solver = Solver(config)

    if config['load_model']:
        solver.load_model(config['load_model_path'])

    if args.train:
        state_dict, cer = solver.sup_train()

    if args.test and args.train:
        solver.test(state_dict=state_dict)
    elif args.test:
        solver.test()
