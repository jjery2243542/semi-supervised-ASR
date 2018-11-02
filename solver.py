import torch 
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from model import E2E, Judge
from dataloader import get_data_loader
from dataset import PickleDataset, NegativeDataset
from utils import *
import yaml
import os
import pickle

class Solver(object):
    def __init__(self, config, load_model=False):

        self.config = config
        print(self.config)

        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()
       
        # get data loader
        self.get_data_loaders()

        # get label distribution
        self.get_label_dist(self.train_lab_dataset)

        # calculate proportion between features and characters
        self.proportion = self.calculate_length_proportion()

        # build model and optimizer
        self.build_model(load_model=load_model)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), f'{model_path}.ckpt')
        torch.save(self.gen_opt.state_dict(), f'{model_path}.opt')
        return

    def save_judge(self, model_path):
        torch.save(self.judge.state_dict(), f'{model_path}.judge.ckpt')
        torch.save(self.dis_opt.state_dict(), f'{model_path}.judge.opt')
        return

    def pad_ys_and_distr(self, ys, distr):
        # convert lab_ys to onehot, pad y to longest
        max_length = max(max(y.size(0) for y in ys), distr.size(1))
        pad_ys = pad_list(ys, pad_value=self.vocab['<EOS>'], max_length=max_length)
        pad_ys_onehot = onehot(pad_ys, encode_dim=distr.size(-1))

        # pad distr
        if distr.size(1) < pad_ys_onehot.size(1):
            pad_length = max_length - distr.size(1) 
            pad_vec = onehot(torch.LongTensor([self.vocab['<EOS>']]), encode_dim=distr.size(-1))
            pad_tensor = pad_vec.expand(distr.size(0), pad_length, pad_vec.size(1))
            distr = torch.cat([distr, pad_tensor], dim=1)
        return pad_ys_onehot, distr

    def calc_gradient_penalty(self, lab_enc, lab_enc_len, lab_ys, unlab_enc, unlab_enc_len, unlab_distr):
        batch_size = lab_enc.size(0)
        alpha = cc(torch.rand(batch_size))
        # pad and make ys onehot
        lab_ys_onehot, unlab_distr = self.pad_ys_and_distr(lab_ys, unlab_distr)

        # interpolate
        interpolate_enc = interpolate_with_diff_len(lab_enc, unlab_enc, alpha)
        interpolate_len = [max(a, b) for a, b in zip(lab_enc_len, unlab_enc_len)]
        interpolate_y = interpolate_with_diff_len(lab_ys_onehot, unlab_distr, alpha)

        disc_interpolates, _ = self.judge.scorer(interpolate_enc, interpolate_len, interpolate_y, is_distr=True)

        gradients_x = autograd.grad(outputs=disc_interpolates, inputs=interpolate_enc, 
                grad_outputs=cc(torch.ones(disc_interpolates.size())),
                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients_y = autograd.grad(outputs=disc_interpolates, inputs=interpolate_y, 
                grad_outputs=cc(torch.ones(disc_interpolates.size())),
                create_graph=True, retain_graph=True, only_inputs=True)[0]
        lp_x = (F.relu((gradients_x.norm(2, dim=1) - 1)) ** 2).mean()
        lp_y = (F.relu((gradients_y.norm(2, dim=1) - 1)) ** 2).mean()
        lp = lp_x + lp_y
        return lp

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f)
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f)
        return

    def load_model(self, model_path, load_optimizer):
        print(f'Load model from {model_path}.ckpt')
        self.model.load_state_dict(torch.load(f'{model_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}.opt')
            self.opt.load_state_dict(torch.load(f'{model_path}.opt'))
            adjust_learning_rate(self.opt, self.config['g_learning_rate'])
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

        # get negative sample dataloader for judge training
        self.neg_dataset = NegativeDataset(os.path.join(root_dir, f'{labeled_set}.pkl'),
                config=self.config, sort=True)
        self.neg_loader = get_data_loader(self.neg_dataset, batch_size=self.config['batch_size'], 
                shuffle=True)
        return

    def get_infinite_iter(self):
        # dataloader to cycle iterator 
        self.lab_iter = infinite_iter(self.train_lab_loader)
        self.unlab_iter = infinite_iter(self.train_unlab_loader)
        self.neg_iter = infinite_iter(self.neg_loader)
        return

    def build_model(self, load_model=False):
        labeldist = self.labeldist
        ls_weight = self.config['ls_weight']

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
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], 
                weight_decay=self.config['weight_decay'])
        self.gen_opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], 
                weight_decay=self.config['weight_decay'], betas=(0.5, 0.9))

        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])

        self.judge = cc(
                Judge(encoder=self.model.encoder, 
                    attention=self.model.attention,
                    decoder=self.model.decoder, 
                    input_dim=self.config['input_dim'],
                    enc_hidden_dim=self.config['enc_hidden_dim'],
                    enc_n_layers=self.config['enc_n_layers'],
                    subsample=self.config['subsample'],
                    dropout_rate=self.config['dropout_rate'],
                    dec_hidden_dim=self.config['dec_hidden_dim'],
                    att_dim=self.config['att_dim'],
                    conv_channels=self.config['conv_channels'],
                    conv_kernel_size=self.config['conv_kernel_size'],
                    att_odim=self.config['att_odim'],
                    embedding_dim=self.config['embedding_dim'],
                    output_dim=len(self.vocab),
                    pad=self.vocab['<PAD>'],
                    eos=self.vocab['<EOS>'],
                    shared=self.config['judge_share_param']
                    ))
        print(self.judge)
        # exponential moving average
        self.ema = EMA(momentum=self.config['ema_momentum'])
        if self.config['judge_share_param']:
            self.dis_opt = torch.optim.Adam(self.judge.scorer.parameters(), lr=self.config['d_learning_rate'], 
                weight_decay=self.config['weight_decay'], betas=(0.5, 0.9))
        else:
            self.dis_opt = torch.optim.Adam(self.judge.parameters(), lr=self.config['d_learning_rate'], 
                weight_decay=self.config['weight_decay'], betas=(0.5, 0.9))
        return

    def ind2sent(self, all_prediction, all_ys):
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])
        #prediction_til_eos = all_prediction

        # indexes to characters
        prediction_sents = to_sents(prediction_til_eos, self.vocab, self.non_lang_syms)
        ground_truth_sents = to_sents(all_ys, self.vocab, self.non_lang_syms)

        # calculate cer
        cer = calculate_cer(prediction_sents, ground_truth_sents)
        return cer, prediction_sents, ground_truth_sents

    def validation(self):

        self.model.eval()
        all_prediction, all_ys = [], []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):

            xs, ilens, ys = to_gpu(data)

            # calculate loss
            _, log_probs, _ , _ = self.model(xs, ilens, ys=ys)
            loss = self.model.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # max length in ys
            max_dec_timesteps = max([y.size(0) for y in ys])

            # feed previous
            _, _ , prediction, _ = self.model(xs, ilens, ys=None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()
        # calculate loss
        avg_loss = total_loss / len(self.dev_loader)

        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)

        return avg_loss, cer, prediction_sents, ground_truth_sents

    def test(self, state_dict=None):

        # load model
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
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

            xs, ilens, ys = to_gpu(data)

            # max length in ys
            max_dec_timesteps = max([y.size(0) for y in ys])

            # feed previous
            _, _, prediction, _ = self.model(xs, ilens, ys=None, 
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

    def judge_train_one_iteration(self, 
            lab_xs, 
            lab_ilens, 
            lab_ys,
            unlab_xs, 
            unlab_ilens,
            neg_xs,
            neg_ilens,
            neg_ys):


        # get unlab distr by argmax
        unlab_logits, _, _, _ = self.model(unlab_xs, unlab_ilens, ys=None, sample=False, 
                max_dec_timesteps=int(unlab_xs.size(1) * self.proportion))
        unlab_distr = F.softmax(unlab_logits, dim=-1)

        # using distribution as y_hat
        unlab_scores, _, unlab_enc, unlab_enc_len = self.judge(unlab_xs, unlab_ilens, unlab_distr, is_distr=True)
        lab_scores, _, lab_enc, lab_enc_len = self.judge(lab_xs, lab_ilens, lab_ys)

        # use mismatched (speech, text) for negative samples
        neg_scores, _, _, _ = self.judge(neg_xs, neg_ilens, neg_ys)

        avg_lab_score = torch.mean(lab_scores)
        avg_unlab_score = torch.mean(unlab_scores)
        avg_neg_score = torch.mean(neg_scores)
        w_distance = avg_lab_score - (avg_unlab_score + avg_neg_score) / 2
        gp = self.calc_gradient_penalty(lab_enc, lab_enc_len, lab_ys, unlab_enc, unlab_enc_len, unlab_distr) 
        loss = -w_distance + self.config['lambda_gp'] * gp

        # calculate gradients

        self.dis_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.judge.parameters(), max_norm=self.config['max_grad_norm'])
        self.dis_opt.step()

        meta = {'w_distance':w_distance.item(),
                'real_score':avg_lab_score.item(),
                'fake_score':avg_unlab_score.item(),
                'neg_score':avg_neg_score.item(),
                'gradient_penalty':gp.item()}
        return meta

    def judge_pretrain(self):
        # using cycle-iterator to get data
        self.get_infinite_iter()
        judge_iterations = self.config['judge_iterations']
        print('--------Judge pretraining--------')
        for iteration in range(judge_iterations):
            # load data
            lab_data = next(self.lab_iter)
            unlab_data = next(self.unlab_iter)
            neg_data = next(self.neg_iter)

            lab_xs, lab_ilens, lab_ys = to_gpu(lab_data)
            unlab_xs, unlab_ilens, _ = to_gpu(unlab_data)
            neg_xs, neg_ilens, neg_ys = to_gpu(neg_data)

            meta = self.judge_train_one_iteration(
                    lab_xs, lab_ilens, lab_ys, 
                    unlab_xs, unlab_ilens,
                    neg_xs, neg_ilens, neg_ys)

            w_distance = meta['w_distance']

            real_score = meta['real_score']
            fake_score = meta['fake_score']
            neg_score = meta['neg_score']

            gp = meta['gradient_penalty']

            print(f'Iter:[{iteration + 1}/{judge_iterations}], '
                    f'w_dis: {w_distance:.3f}, real_score: {real_score:.3f}, fake_score: {fake_score:.3f}'
                    f', neg_score: {neg_score:.3f}, gp: {gp:.2f}', end='\r')

            # add to tensorboard
            tag = self.config['tag']
            for key, val in meta.items():
                self.logger.scalar_summary(f'{tag}/judge_pretrain/{key}', val, iteration + 1)

            if (iteration + 1) % self.config['summary_steps'] == 0 or (iteration + 1) == judge_iterations:
                print('')
                model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
                self.save_judge(model_path)
        return 
    
    def sup_train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_lab_loader)
        total_loss = 0.

        for train_steps, data in enumerate(self.train_lab_loader):

            xs, ilens, ys = to_gpu(data)

            # add gaussian noise after gaussian_epoch
            if self.config['add_gaussian'] and epoch >= self.config['gaussian_epoch']:
                gau = np.random.normal(0, self.config['gaussian_std'], (xs.size(0), xs.size(1), xs.size(2)))
                gau = cc(torch.from_numpy(np.array(gau, dtype=np.float32)))
                xs = xs + gau

            # input the model
            _, log_probs, prediction, _ = self.model(xs, ilens, ys, tf_rate=tf_rate, sample=False)
            # mask and calculate loss
            loss = -torch.mean(log_probs)
            #loss = self.model.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # calculate gradients 
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
            self.opt.step()
            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {loss:.3f}', end='\r')
            # add to logger
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train_loss', value=loss.item(), 
                    step=epoch * total_steps + train_steps + 1)

        return total_loss / total_steps

    def sup_pretrain(self):

        best_cer = 200
        best_model = None

        # tf_rate
        init_tf_rate = self.config['init_tf_rate']
        tf_decay_epochs = self.config['tf_decay_epochs']
        tf_rate_lowerbound = self.config['tf_rate_lowerbound']

	# lr scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, 
                milestones=[self.config['change_learning_rate_epoch']],
                gamma=self.config['lr_gamma'])

        print('------supervised pretraining-------')

        for epoch in range(self.config['epochs']):
            # schedule
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
            # save model in every epoch
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:03d}')

        return best_model, best_cer

    def gen_train_one_iteration(self, 
            lab_xs, lab_ilens, lab_ys,
            unlab_xs, unlab_ilens):

        unlab_logits, _, _, _ = self.model(unlab_xs, unlab_ilens, ys=None, sample=False, 
                max_dec_timesteps=int(unlab_xs.size(1) * self.proportion))
        unlab_distr = F.softmax(unlab_logits, dim=-1)

        # using distribution as y_hat
        unlab_scores, _, _, _ = self.judge(unlab_xs, unlab_ilens, unlab_distr, is_distr=True)
        unsup_loss = -torch.mean(unlab_scores)

        # mask and calculate loss
        _, lab_log_probs, _, _ = self.model(lab_xs, lab_ilens, ys=lab_ys, tf_rate=1.0, sample=False)
        sup_loss = -torch.mean(lab_log_probs)
        gen_loss = sup_loss + self.config['unsup_weight'] * unsup_loss

        # calculate gradients 

        self.gen_opt.zero_grad()
        gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
        self.gen_opt.step()

        gen_meta = {'unsup_loss': unsup_loss.item(),
                    'sup_loss': sup_loss.item(),
                    'gen_loss': gen_loss.item()}
        return gen_meta

    def ssl_train_one_iteration(self, iteration): 
        d_steps = self.config['d_steps']
        g_steps = self.config['g_steps']

        # load data
        lab_data = next(self.lab_iter)
        neg_data = next(self.neg_iter)
        unlab_data = next(self.unlab_iter)

        lab_xs, lab_ilens, lab_ys = to_gpu(lab_data)
        neg_xs, neg_ilens, neg_ys = to_gpu(neg_data)
        unlab_xs, unlab_ilens, _ = to_gpu(unlab_data)

        # train D steps of discriminator
        for d_step in range(d_steps):
            dis_meta = self.judge_train_one_iteration(
                    lab_xs, lab_ilens, lab_ys, 
                    unlab_xs, unlab_ilens,
                    neg_xs, neg_ilens, neg_ys)

            w_distance = dis_meta['w_distance']

            real_score = dis_meta['real_score']
            fake_score = dis_meta['fake_score']
            neg_score = dis_meta['neg_score']

            gp = dis_meta['gradient_penalty']

            print(f'Dis:[{d_step + 1}/{d_steps}], '
                    f'w_dis: {w_distance:.3f}, real_score: {real_score:.3f}, fake_score: {fake_score:.3f}'
                    f', neg_score: {neg_score:.3f}, gp: {gp:.2f}', end='\r')

            # add to tensorboard
            step = iteration * d_steps + d_step + 1
            tag = self.config['tag']
            for key, val in dis_meta.items():
                self.logger.scalar_summary(f'{tag}/ssl_judge/{key}', val, step)
        print()

        # train G step of generator
        for g_step in range(g_steps):
            gen_meta = self.gen_train_one_iteration(
                    lab_xs, lab_ilens, lab_ys,
                    unlab_xs, unlab_ilens)

            unsup_loss = gen_meta['unsup_loss']
            sup_loss = gen_meta['sup_loss']
            gen_loss = gen_meta['gen_loss']

            print(f'Gen:[{g_step + 1}/{g_steps}], '
                    f'sup_loss: {sup_loss:.3f}, unsup_loss: {unsup_loss:.3f}, gen_loss: {gen_loss:.3f}',
                    end='\r')

            # add to tensorboard
            step = iteration * g_steps + g_step + 1
            tag = self.config['tag']
            for key, val in gen_meta.items():
                self.logger.scalar_summary(f'{tag}/ssl_generator/{key}', val, step + 1)
        print()

        if d_steps > 0:
            meta = {**dis_meta, **gen_meta}
            return meta 

    def ssl_train(self):
        print('--------SSL training--------')
        # adjust learning rate
        adjust_learning_rate(self.gen_opt, self.config['g_learning_rate'])
        print(self.gen_opt)

        best_cer = 1000
        best_model = None

        total_steps = self.config['ssl_iterations']

        if not hasattr(self, 'lab_iter'):
            self.get_infinite_iter()

        for step in range(total_steps):
            meta = self.ssl_train_one_iteration(iteration=step)

            if (step + 1) % self.config['summary_steps'] == 0 or step + 1 == total_steps:
                avg_valid_loss, cer, prediction_sents, ground_truth_sents = self.validation()

                print(f'Iter: [{step + 1}/{total_steps}], valid_loss={avg_valid_loss:.4f}, CER={cer:.4f}')

                # add to tensorboard
                tag = self.config['tag']
                self.logger.scalar_summary(f'{tag}/ssl/cer', cer, step + 1)
                self.logger.scalar_summary(f'{tag}/ssl/val_loss', avg_valid_loss, step + 1)

                # only add first n samples
                lead_n = 5
                print('-----------------')
                for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                    self.logger.text_summary(f'{tag}/ssl/prediction-{i}', p, step + 1)
                    self.logger.text_summary(f'{tag}/ssl/ground_truth-{i}', gt, step + 1)
                    print(f'hyp-{i+1}: {p}')
                    print(f'ref-{i+1}: {gt}')
                print('-----------------')

                if cer < best_cer: 
                    # save model 
                    model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
                    best_cer = cer
                    self.save_model(model_path)
                    self.save_judge(model_path)

                    best_model = self.model.state_dict()
                    print(f'Save #{step} model, val_loss={avg_valid_loss:.3f}, CER={cer:.3f}')
                    print('-----------------')
        
if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
