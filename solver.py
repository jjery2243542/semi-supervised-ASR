import torch 
import numpy as np
from model import E2E, Judge
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
import yaml
import os
import pickle

class Solver(object):
    def __init__(self, config, mode='train'):

        self.mode = mode
        self.config = config
        print(self.config)

        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()
       
        # load data loaders only in training mode
        if mode == 'train':
            # get data loader
            self.get_data_loaders()

            # get label distribution
            self.get_label_dist(self.train_lab_dataset)

            # calculate proportion between features and characters
            self.proportion = self.calculate_length_proportion()

        # build model and optimizer
        self.build_model(mode=mode)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), f'{model_path}.ckpt')
        torch.save(self.gen_opt.state_dict(), f'{model_path}.opt')
        return

    def save_judge(self, model_path):
        torch.save(self.judge.state_dict(), f'{model_path}.judge.ckpt')
        torch.save(self.dis_opt.state_dict(), f'{model_path}.judge.opt')
        return 

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f)
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f)
        return

    def load_model(self, model_path, load_optimizer):
        self.model.load_state_dict(torch.load(f'{model_path}.ckpt'))
        if load_optimizer:
            self.gen_opt.load_state_dict(torch.load(f'{model_path}.opt'))
        return

    def load_judge(self, model_path, load_optmizer):
        self.judge.load_state_dict(torch.load(f'{model_path}.judge.ckpt'))
        if load_optimizer:
            self.dis_opt.load_state_dict(torch.load(f'{model_path}.judge.opt'))
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

    def calculate_length_proportion(self):
        x_len, y_len = 0, 0
        for x, y in self.train_lab_dataset:
            x_len += x.shape[0]
            y_len += len(y)
        return y_len / x_len
             
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

    def build_model(self, mode):
        if mode == 'train':
            labeldist = self.labeldist
            ls_weight = self.config['ls_weight']
        else:
            labeldist = None
            ls_weight = 0

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
            ls_weight=ls_weight,
            labeldist=labeldist,
            pad=self.vocab['<PAD>'],
            bos=self.vocab['<BOS>'],
            eos=self.vocab['<EOS>']
            ))
        print(self.model)

        # build judge model only when training mode
        if mode == 'train':
            self.judge = cc(Judge(dropout_rate=self.config['dropout_rate'],
                dec_hidden_dim=self.config['dec_hidden_dim'],
                att_odim=self.config['att_odim'],
                embedding_dim=self.config['embedding_dim'],
                output_dim=len(self.vocab),
                encoder=self.model.encoder,
                attention=self.model.attention,
                pad=self.vocab['<PAD>'],
                eos=self.vocab['<EOS>'],
                shared=self.config['judge_share_param']
                ))
            print(self.judge)

        self.gen_opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], 
                weight_decay=self.config['weight_decay'])
        if self.config['judge_share_param']:
            self.dis_opt = torch.optim.Adam(self.judge.scorer.parameters(), lr=self.config['d_learning_rate']) 
        else:
            self.dis_opt = torch.optim.Adam(self.judge.parameters(), lr=self.config['d_learning_rate'])
        return

    def judge_train_one_iteration(self, lab_data_iterator, unlab_data_iterator):
        # load data
        lab_data, unlab_data = next(lab_data_iterator), next(unlab_data_iterator)
        lab_xs, lab_ilens, lab_ys = self.to_gpu(lab_data)
        unlab_xs, unlab_ilens, _ = self.to_gpu(unlab_data)

        # TODO:greedy decode now, change to TOPK decode
        
        _, unlab_ys_hat, _ = self.model(unlab_xs, unlab_ilens, ys=None, 
                max_dec_timesteps=lab_xs.size(1) * self.proportion + self.config['extra_length'])

        unlab_ys_hat = remove_pad_eos(unlab_yhat, eos=self.vocab['<EOS>'])

        lab_probs = self.judge(lab_xs, lab_ilens, lab_ys)
        real_labels = cc(torch.ones(lab_probs.size(0)))
        real_loss, real_probs = self.judge.mask_and_cal_loss(lab_probs, lab_ys, target=real_labels)
        real_correct = torch.sum((real_probs >= 0.5).float())

        unlab_probs = self.judge(unlab_xs, unlab_ilens, unlab_ys_hat)
        fake_labels = cc(torch.zeros(unlab_probs.size(0)))
        fake_loss, fake_probs = self.judge.mask_and_cal_loss(unlab_probs, unlab_ys_hat, target=fake_labels)
        fake_correct = torch.sum((fake_probs < 0.5).float())

        loss = real_loss + fake_loss
        acc = (real_correct + fake_correct) / (lab_probs.size(0) + unlab_probs.size(0))

        # calculate gradients 
        self.dis_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.judge.parameters(), max_norm=self.config['max_grad_norm'])
        self.dis_opt.step()

        return loss.item(), acc.item()

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
            loss = self.model.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # calculate gradients 
            self.gen_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
            self.gen_opt.step()

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

    def test(self, state_dict=None):

        # load model
        if not state_dict:
            self.load_model(self.config['load_model_path'])
        else:
            self.model.load_state_dict(state_dict)

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']

        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.pkl'), 
            config=None, sort=False)

        test_loader = get_data_loader(test_dataset, 
                batch_size=1, 
                shuffle=False)

        self.model.eval()
        all_prediction, all_ys = [], []

        for step, data in enumerate(test_loader):

            xs, ilens, ys = self.to_gpu(data)

            # max length in ys
            max_dec_timesteps = max([y.size(0) for y in ys])

            # feed previous
            _ , prediction, _ = self.model(xs, ilens, ys=None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()

        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)

        with open(f'{test_set}.txt', 'w') as f:
            for p in prediction_sents:
                f.write(f'{p}\n')

        print(f'{test_set}: {len(prediction_sents)} utterances, CER={cer:.4f}')
        return cer

    def validation(self):

        self.model.eval()
        all_prediction, all_ys = [], []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):

            xs, ilens, ys = self.to_gpu(data)

            # calculate loss
            log_probs, _ , _ = self.model(xs, ilens, ys=ys)
            loss = self.model.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # max length in ys
            max_dec_timesteps = max([y.size(0) for y in ys])

            # feed previous
            _ , prediction, _ = self.model(xs, ilens, ys=None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()
        # calculate loss
        avg_loss = total_loss / len(self.dev_loader)

        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)

        return avg_loss, cer, prediction_sents, ground_truth_sents

    def judge_pretrain(self):

        best_model = None

        # dataloader to cycle iterator 
        lab_iter = infinite_iter(self.train_lab_loader)
        unlab_iter = infinite_iter(self.train_lab_loader)

        # adjust learning rate
        adjust_learning_rate(self.gen_opt, self.config['g_learning_rate'])

        # set self.model to eval mode
        self.model.eval()

        total_loss = 0.
        judge_iterations = self.config['judge_iterations']
        for iteration in range(judge_iterations):
            loss, acc = self.judge_train_one_iteration(lab_iter, unlab_iter)
            total_loss += loss

            print(f'Iter:[{iteration + 1}/{judge_iterations}], loss: {loss:.3f}, acc: {acc:.3f}', end='\r')

            if iteration + 1 % self.config['summary_steps']:
                # add to tensorboard
                tag = self.config['tag']
                self.logger.scalar_summary(f'{tag}/judge_pretrain/train_loss', loss, iteration + 1)
                self.logger.scalar_summary(f'{tag}/judge_pretrain/acc', acc, iteration + 1)
                self.save_judge()
        return 

    def ssl_train_one_iteration(self, lab_data_iterator, unlab_data_iterator):
        # load data
        lab_data, unlab_data = next(lab_data_iterator), next(unlab_data_iterator)
        lab_xs, lab_ilens, lab_ys = self.to_gpu(lab_data)
        unlab_xs, unlab_ilens, _ = self.to_gpu(unlab_data)

        # TODO:greedy decode now, change to TOPK decode
        
        _, unlab_ys_hat, _ = self.model(unlab_xs, unlab_ilens, ys=None, 
                max_dec_timesteps=lab_xs.size(1) * self.proportion + self.config['extra_length'])

        unlab_ys_hat = remove_pad_eos(unlab_yhat, eos=self.vocab['<EOS>'])

        lab_probs = self.judge(lab_xs, lab_ilens, lab_ys)
        real_labels = cc(torch.ones(lab_probs.size(0)))
        real_loss, real_probs = self.judge.mask_and_cal_loss(lab_probs, lab_ys, target=real_labels)
        real_correct = torch.sum((real_probs >= 0.5).float())

        unlab_probs = self.judge(unlab_xs, unlab_ilens, unlab_ys_hat)
        fake_labels = cc(torch.zeros(unlab_probs.size(0)))
        fake_loss, fake_probs = self.judge.mask_and_cal_loss(unlab_probs, unlab_ys_hat, target=fake_labels)
        fake_correct = torch.sum((fake_probs < 0.5).float())

        loss = real_loss + fake_loss
        acc = (real_correct + fake_correct) / (lab_probs.size(0) + unlab_probs.size(0))

        # calculate gradients 
        self.dis_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.judge.parameters(), max_norm=self.config['max_grad_norm'])
        self.dis_opt.step()

        return loss.item(), acc.item()
        

    def ssl_train(self):
        pass

        
    def sup_pretrain(self):

        best_cer = 2
        best_model = None

        # lr scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_opt, 
                milestones=[self.config['change_learning_rate_epoch']],
                gamma=self.config['lr_gamma'])

        # tf_rate
        init_tf_rate = self.config['init_tf_rate']
        tf_decay_epochs = self.config['tf_decay_epochs']
        tf_rate_lowerbound = self.config['tf_rate_lowerbound']

        for epoch in range(self.config['epochs']):

            scheduler.step()

            # calculate tf rate
            if epoch <= tf_decay_epochs:
                tf_rate = init_tf_rate - (init_tf_rate - tf_rate_lowerbound) * (epoch / tf_decay_epochs)
            else:
                tf_rate = tf_rate_lowerbound

            # train one epoch
            avg_train_loss = self.sup_train_one_epoch(epoch, tf_rate)

            # validation
            avg_valid_loss, cer, prediction_sents, ground_truth_sents = self.validation()

            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_loss={avg_train_loss:.4f}, '
                    f'valid_loss={avg_valid_loss:.4f}, CER={cer:.4f}')

            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/supervised/cer', cer, epoch)
            self.logger.scalar_summary(f'{tag}/supervised/val_loss', avg_valid_loss, epoch)

            # only add first n samples
            lead_n = 5
            print('-----------------')
            for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                self.logger.text_summary(f'{tag}/supervised/prediction-{i}', p, epoch)
                self.logger.text_summary(f'{tag}/supervised/ground_truth-{i}', gt, epoch)
                print(f'hyp-{i+1}: {p}')
                print(f'ref-{i+1}: {gt}')
            print('-----------------')

            if cer < best_cer: 
                # save model 
                model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
                best_cer = cer
                self.save_model(model_path)
                best_model = self.model.state_dict()
                print(f'Save #{epoch} model, val_loss={avg_valid_loss:.3f}, CER={cer:.3f}')
                print('-----------------')

        return best_model, best_cer

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
    #solver.sup_train()
