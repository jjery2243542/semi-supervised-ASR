import torch 
import numpy as np
from model import E2E
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import cc, _seq_mask, remove_pad_eos, ind2character, calculate_cer, char_list_to_str, to_sents, Logger
import yaml
import os
import pickle

class Solver(object):
    def __init__(self, config):
        self.config = config
        print(self.config)

        # store model 
        self.model_kept = []

        #self.max_kept = config['max_kept']

        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()

        # get data loader 
        self.get_data_loaders()

        # get label distribution
        self.get_label_dist(self.train_lab_dataset)

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

    def get_label_dist(self, dataset):
        labelcount = np.zeros(len(self.vocab))
        for _, y in dataset:
            for ind in y:
                labelcount[ind] += 1.
        labelcount[self.vocab['<EOS>']] += len(dataset)
        labelcount[self.vocab['<PAD>']] = 0
        labelcount[self.vocab['<BOS>']] = 0
        self.labeldist = labelcount / np.sum(labelcount)
        return
             
    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        # get labeled dataset
        labeled_set = self.config['labeled_set']
        self.train_lab_dataset = PickleDataset(os.path.join(root_dir, f'{labeled_set}.pkl'), 
            config=self.config, sort=True)
        self.train_lab_loader = get_data_loader(self.train_lab_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get unlabeled dataset
        unlabeled_set = self.config['unlabeled_set']
        self.train_unlab_dataset = PickleDataset(os.path.join(root_dir, f'{unlabeled_set}.pkl'), 
            config=self.config, sort=True)
        self.train_unlab_loader = get_data_loader(self.train_unlab_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get dev dataset
        dev_set = self.config['dev_set']
        # do not sort dev set
        self.dev_dataset = PickleDataset(os.path.join(root_dir, f'{dev_set}.pkl'), sort=True)
        self.dev_loader = get_data_loader(self.dev_dataset, 
                batch_size=self.config['batch_size'] // 2, 
                shuffle=False)
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
            ls_weight=self.config['ls_weight'],
            labeldist=self.labeldist,
            pad=self.vocab['<PAD>'],
            bos=self.vocab['<BOS>'],
            eos=self.vocab['<EOS>']
            ))
        print(self.model)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], 
                weight_decay=self.config['weight_decay'])
        return

    def mask_and_cal_loss(self, log_prob, ys):
        # add 1 to EOS
        seq_len = [y.size(0) + 1 for y in ys]
        mask = cc(_seq_mask(seq_len=seq_len, max_len=log_prob.size(1)))
        # divide by total length
        loss = -torch.sum(log_prob * mask) / sum(seq_len)
        return loss

    def sup_train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_lab_loader)
        total_loss = 0.

        for train_steps, data in enumerate(self.train_lab_loader):

            xs, ilens, ys = self.to_gpu(data)

            # add gaussian noise after gaussian_epoch
            if self.config['add_gaussian'] and epoch >= self.config['gaussian_epoch']:
                gau = np.random.normal(0, self.config['gaussian_std'], (xs.size(0), xs.size(1), xs.size(2)))
                gau = cc(torch.from_numpy(np.array(gau, dtype=np.float32)))
                xs = xs + gau

            # input the model
            log_probs, prediction, _ = self.model(xs, ilens, ys, tf_rate=tf_rate)

            # mask and calculate loss
            loss = self.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # calculate gradients 
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
            self.opt.step()

            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {loss:.3f}', end='\r')
            if (train_steps + 1) % self.config['summary_steps'] == 0:
                # add to logger
                tag = self.config['tag']
                self.logger.scalar_summary(tag=f'{tag}/train_loss', value=loss.item(), 
                        step=epoch * total_steps + train_steps + 1)

        return total_loss / total_steps

    def to_gpu(self, data):
        xs, ilens, ys = data
        xs = cc(xs)
        ys = [cc(y) for y in ys]
        return xs, ilens, ys

    def ind2sent(self, all_prediction, all_ys):
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])

        # indexes to characters
        prediction_sents = to_sents(prediction_til_eos, self.vocab, self.non_lang_syms)
        ground_truth_sents = to_sents(all_ys, self.vocab, self.non_lang_syms)

        # calculate cer
        cer = calculate_cer(prediction_sents, ground_truth_sents)
        return cer, prediction_sents, ground_truth_sents

    def test(self):

        # load model
        self.load_model(self.config['load_model_path'])

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']

        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.pkl'), 
            config=None, sort=True)

        test_loader = get_data_loader(test_dataset, 
                batch_size=self.config['batch_size'] // 2, 
                shuffle=False)

        self.model.eval()
        all_prediction, all_ys = [], []

        for step, data in enumerate(test_loader):

            xs, ilens, ys = self.to_gpu(data)

            # max length in ys
            max_dec_timesteps = max([y.size(0) for y in ys])

            # feed previous
            _ , prediction, _ = self.model(xs, ilens, ys=None, max_dec_timesteps=max_dec_timesteps + 30)

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()

        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)
        print(f'{test_set}: {len(prediction_sents)} utterances, CER={cer}')

        return cer

    def validation(self):

        self.model.eval()
        all_prediction, all_ys = [], []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):

            xs, ilens, ys = self.to_gpu(data)

            # calculate loss
            log_probs, _ , _ = self.model(xs, ilens, ys=ys)
            loss = self.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # max length in ys
            max_dec_timesteps = max([y.size(0) for y in ys])

            # feed previous
            _ , prediction, _ = self.model(xs, ilens, ys=None, max_dec_timesteps=max_dec_timesteps + 15)

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()
        # calculate loss
        avg_loss = total_loss / len(self.dev_loader)

        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)

        return avg_loss, cer, prediction_sents, ground_truth_sents

    def sup_train(self):

        best_cer = 2

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, 
                milestones=[self.config['change_learning_rate_epoch']],
                gamma=self.config['lr_gamma'])

        # tf_rate
        init_tf_rate = self.config['init_tf_rate']
        tf_decay_epochs = self.config['tf_decay_epochs']
        tf_rate_lowerbound = self.config['tf_rate_lowerbound']

        for epoch in range(self.config['epochs']):

            # lr scheduler
            scheduler.step()

            # calculate tf rate
            tf_rate = init_tf_rate - (init_tf_rate - tf_rate_lowerbound) * (epoch / tf_decay_epochs)

            # train one epoch
            avg_train_loss = self.sup_train_one_epoch(epoch, tf_rate)

            # validation
            avg_valid_loss, cer, prediction_sents, ground_truth_sents = self.validation()

            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_loss={avg_train_loss:.4f}, '
                    f'valid_loss={avg_valid_loss:.4f}, CER={cer:.4f}')

            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/cer', cer, epoch)
            self.logger.scalar_summary(f'{tag}/val_loss', avg_valid_loss, epoch)

            # only add first n samples
            lead_n = 5
            print('-----------------')
            for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                self.logger.text_summary(f'prediction-{i}', p, epoch)
                self.logger.text_summary(f'ground_truth-{i}', gt, epoch)
                print(f'hyp-{i+1}: {p}')
                print(f'ref-{i+1}: {gt}')
            print('-----------------')

            if cer < best_cer: 
                # save model 
                model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
                best_cer = cer
                self.save_model(model_path)
                print(f'Save #{epoch} model, val_loss={avg_valid_loss:.3f}, CER={cer:.3f}')
                print('-----------------')


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
    #solver.sup_train()
