import torch 
import numpy as np
from model import E2E
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import cc, _seq_mask, remove_pad_eos, ind2character, calculate_cer, char_list_to_str, Logger
import yaml
import os
import pickle

class Solver(object):
    def __init__(self, config):
        self.config = config

        # store model 
        self.model_kept = []
        self.max_kept = config['max_kept']

        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()

        # get data loader 
        self.get_data_loaders()

        # build model and optimizer
        self.build_model()

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)
        #self.model_kept.append(model_path)
        #if len(self.model_kept) > self.max_kept:
        #    os.remove(self.model_kept[0])
        #    self.model_kept.pop(0)
        return

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f)
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f)
        return

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        return 

    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        # get labeled dataset
        labeled_set = self.config['labeled_set']
        train_lab_dataset = PickleDataset(os.path.join(root_dir, f'{labeled_set}.pkl'), 
            config=self.config, sort=True)
        self.train_lab_loader = get_data_loader(train_lab_dataset, batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get unlabeled dataset
        unlabeled_set = self.config['unlabeled_set']
        train_unlab_dataset = PickleDataset(os.path.join(root_dir, f'{unlabeled_set}.pkl'), 
            config=self.config, sort=True)
        self.train_unlab_loader = get_data_loader(train_unlab_dataset, batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get dev dataset
        dev_set = self.config['dev_set']
        # do not sort dev set
        dev_dataset = PickleDataset(os.path.join(root_dir, f'{dev_set}.pkl'), config=self.config, sort=True)
        self.dev_loader = get_data_loader(dev_dataset, batch_size=self.config['batch_size'] // 2, shuffle=False)
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
            output_dim=len(self.vocab),
            embedding_dim=self.config['embedding_dim'],
            heads=self.config['heads'],
            pad=self.vocab['<PAD>'],
            bos=self.vocab['<BOS>'],
            eos=self.vocab['<EOS>']
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
        total_loss = 0.
        for train_steps, data in enumerate(self.train_lab_loader):
            xs, ilens, ys = data
            # transfer to cuda 
            xs = cc(xs)
            ys = [cc(y) for y in ys]
            # input the model
            log_probs, prediction, _ = self.model(xs, ilens, ys)

            # mask and calculate loss
            loss = self.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # calculate gradients 
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {loss:.3f}', end='\r')
            if (train_steps + 1) % self.config['summary_steps'] == 0:
                # add to logger
                self.logger.scalar_summary(tag=self.config['tag'], value=loss.item(), 
                        step=epoch * total_steps + train_steps + 1)
        return total_loss / total_steps

    def validation(self):
        self.model.eval()
        all_prediction, all_ys = [], []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):
            xs, ilens, ys = data
            xs = cc(xs)
            ys = [cc(y) for y in ys]
            # max length in ys
            max_dec_timesteps = max([y.size(0) for y in ys])
            # calculate loss
            log_probs, _ , _ = self.model(xs, ilens, ys=ys, max_dec_timesteps=max_dec_timesteps)
            loss = self.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()
            # feed previous
            _ , prediction, _ = self.model(xs, ilens, ys=None, max_dec_timesteps=max_dec_timesteps)
            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]
        self.model.train()
        # calculate loss
        avg_loss = total_loss / len(self.dev_loader)
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])
        # indexes to characters
        prediction_char = ind2character(prediction_til_eos, self.non_lang_syms, self.vocab) 
        ground_truth_char = ind2character(all_ys, self.non_lang_syms, self.vocab)
        prediction_sents = char_list_to_str(prediction_char) 
        ground_truth_sents = char_list_to_str(ground_truth_char)
        cer = calculate_cer(prediction_sents, ground_truth_sents)
        return avg_loss, cer, prediction_sents, ground_truth_sents

    def sup_train(self):
        best_valid_loss = 100
        for epoch in range(self.config['epochs']):
            avg_train_loss = self.sup_train_one_epoch(epoch)
            # validation
            avg_valid_loss, cer, prediction_sents, ground_truth_sents = self.validation()
            print(f'epoch: {epoch}, train_loss={avg_train_loss:.4f}, '
                    f'valid_loss={avg_valid_loss:.4f}, CER={cer:.4f}')
            # add to tensorboard
            self.logger.scalar_summary('cer', cer, epoch)
            # only add first 3 samples
            lead_n = 3
            for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                self.logger.text_summary(f'prediction-{i}', p, epoch)
                self.logger.text_summary(f'ground_truth-{i}', gt, epoch)
                print(f'hyp-{i+1}: {p}')
                print(f'ref-{i+1}: {gt}')
            # save model 
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            if avg_valid_loss < best_valid_loss: 
                best_valid_loss = avg_valid_loss
                self.save_model(model_path)

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
    solver.sup_train()
