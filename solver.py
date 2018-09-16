import torch 
import numpy as np
from model import E2E
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import cc, _seq_mask, Logger
import yaml
import os

class Solver(object):
    def __init__(self, config):
        self.config = config

        # store model 
        self.model_kept = []
        self.max_kept = config['max_kept']

        # logger
        self.logger = Logger(config['logdir'])

        # get data loader 
        self.get_data_loaders()

        # build model and optimizer
        self.build_model()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        self.model_kept.append(model_path)
        if len(self.model_kept) > self.max_kept:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)
        return

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        return 

    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        # get labeled dataset
        labeled_set = self.config['labeled_set']
        train_lab_dataset = PickleDataset(os.path.join(root_dir, f'{labeled_set}.pkl'), config=self.config)
        self.train_lab_loader = get_data_loader(train_lab_dataset, batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get unlabeled dataset
        unlabeled_set = self.config['unlabeled_set']
        train_unlab_dataset = PickleDataset(os.path.join(root_dir, f'{unlabeled_set}.pkl'), config=self.config)
        self.train_unlab_loader = get_data_loader(train_unlab_dataset, batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get dev dataset
        dev_set = self.config['dev_set']
        # do not sort dev set
        dev_dataset = PickleDataset(os.path.join(root_dir, f'{dev_set}.pkl'), config=self.config, sort=False)
        self.dev_loader = get_data_loader(dev_dataset, batch_size=self.config['batch_size'], shuffle=False)
        return

    def build_model(self):
        self.model = cc(E2E(input_dim=self.config['input_dim'],
            enc_hidden_dim=self.config['enc_hidden_dim'],
            enc_n_layers=self.config['enc_n_layers'],
            subsample=self.config['subsample'],
            dropout_rate=self.config['dropout_rate'],
            dec_hidden_dim=self.config['dec_hidden_dim'],
            att_dim=self.config['att_dim'],
            conv_channels=self.config['conv_channels'],
            conv_kernel_size=self.config['conv_kernel_size'],
            att_odim=self.config['att_odim'],
            output_dim=self.config['output_dim'],
            heads=self.config['heads']
            ))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], amsgrad=True)
        return

    def mask_and_cal_loss(self, log_prob, ys):
        # add 1 to EOS
        seq_len = [y.size(0) + 1 for y in ys]
        mask = cc(_seq_mask(seq_len=seq_len, max_len=log_prob.size(1)))
        # divide by positive value
        loss = -torch.sum(log_prob * mask) / torch.sum(mask)
        return loss

    def sup_train_one_epoch(self, epoch):
        total_steps = len(self.train_lab_loader)
        for train_steps, data in enumerate(self.train_lab_loader):
            xs, ilens, ys = data
            # transfer to cuda 
            xs = cc(xs)
            ys = [cc(y) for y in ys]
            # input the model
            log_probs, prediction, ws_list = self.model(xs, ilens, ys)

            # mask and calculate loss
            loss = self.mask_and_cal_loss(log_probs, ys)

            # calculate gradients 
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # print message
            print(f'[{train_steps + 1}/{total_steps}], loss: {loss:.3f}')

            if (train_steps + 1) % self.config['summary_steps'] == 0:
                # add to logger
                self.logger.scalar_summary(tag=self.config['tag'], value=loss.item(), 
                        step=epoch * total_steps + train_steps + 1)
        return

    def sup_train(self):
        for epoch in range(self.config['n_epochs']):
            self.sup_train_one_epoch(epoch)
            # validation

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
    solver.sup_train_one_epoch(0)
