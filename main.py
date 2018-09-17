from solver import Solver
import yaml
from argparse import ArgumentParser
import sys

if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
    solver.sup_train()
