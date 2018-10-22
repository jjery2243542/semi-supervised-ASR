import torch 
import torch.nn.functional as F
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

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

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
            self.gen_opt.load_state_dict(torch.load(f'{model_path}.opt'))
            adjust_learning_rate(self.gen_opt, self.config['g_learning_rate'])
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
        self.gen_opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], 
                weight_decay=self.config['weight_decay'])

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
                weight_decay=self.config['weight_decay'])
        else:
            self.dis_opt = torch.optim.Adam(self.judge.parameters(), lr=self.config['d_learning_rate'], 
                weight_decay=self.config['weight_decay'])
        return

    def ind2sent(self, all_prediction, all_ys):
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])

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

    def sample_and_calculate_judge_probs(self, unlab_xs, unlab_ilens):
        # random sample with average length
        gen_log_probs, unlab_ys_hat, _ = self.model(unlab_xs, unlab_ilens, ys=None, sample=True, 
                max_dec_timesteps=int(unlab_xs.size(1) * self.proportion + self.config['extra_length']))

        # remove tokens after eos 
        unlab_ys_hat = remove_pad_eos_batch(unlab_ys_hat, eos=self.vocab['<EOS>'])
        unlab_probs, _ = self.judge(unlab_xs, unlab_ilens, unlab_ys_hat)
        return unlab_probs, unlab_ys_hat, gen_log_probs

    def judge_train_one_iteration(self, 
            lab_xs, 
            lab_ilens, 
            lab_ys,
            unlab_xs, 
            unlab_ilens,
            neg_xs,
            neg_ilens,
            neg_ys):

        unlab_probs, unlab_ys_hat, _ = self.sample_and_calculate_judge_probs(unlab_xs, unlab_ilens)
        lab_probs, _ = self.judge(lab_xs, lab_ilens, lab_ys)

        # use mismatched (speech, text) for negative samples
        neg_probs, _ = self.judge(neg_xs, neg_ilens, neg_ys)

        # calculate loss and acc
        real_labels = cc(torch.ones(lab_probs.size(0)))
        real_probs, _, _ = self.judge.mask_and_average(lab_probs, lab_ys)
        real_loss = F.binary_cross_entropy(real_probs, real_labels)
        real_correct = torch.sum((real_probs >= 0.5).float())

        fake_labels = cc(torch.zeros(unlab_probs.size(0)))
        fake_probs, _, _ = self.judge.mask_and_average(unlab_probs, unlab_ys_hat)
        fake_loss = F.binary_cross_entropy(fake_probs, fake_labels)
        fake_correct = torch.sum((fake_probs < 0.5).float())

        fake_labels = cc(torch.zeros(neg_probs.size(0)))
        neg_probs, _, _ = self.judge.mask_and_average(neg_probs, neg_ys)
        neg_loss = F.binary_cross_entropy(neg_probs, fake_labels)
        neg_correct = torch.sum((neg_probs < 0.5).float()) 

        loss = real_loss + (fake_loss + neg_loss) / 2
        real_acc = real_correct / lab_probs.size(0)
        fake_acc = fake_correct / unlab_probs.size(0)
        neg_acc = neg_correct / neg_probs.size(0)
        acc = (real_correct + fake_correct + neg_correct) / \
                (lab_probs.size(0) + unlab_probs.size(0) + neg_probs.size(0))
        # calculate gradients

        self.dis_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.judge.parameters(), max_norm=self.config['max_grad_norm'])
        self.dis_opt.step()

        meta = {'real_loss':real_loss.item(),
                'fake_loss':fake_loss.item(),
                'neg_loss':neg_loss.item(),
                'real_acc':real_acc.item(),
                'fake_acc':fake_acc.item(),
                'neg_acc':neg_acc.item(),
                'real_prob':torch.mean(real_probs).item(),
                'fake_prob':torch.mean(fake_probs).item(),
                'neg_prob':torch.mean(neg_probs).item(),
                'loss':loss.item(),
                'acc':acc.item()}
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

            real_loss = meta['real_loss']
            fake_loss = meta['fake_loss']
            neg_loss = meta['neg_loss']

            acc = meta['acc']

            print(f'Iter:[{iteration + 1}/{judge_iterations}], '
                    f'real_loss: {real_loss:.3f}, fake_loss: {fake_loss:.3f}, neg_loss: {neg_loss:.3f}'
                    f', acc: {acc:.2f}', end='\r')

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
            log_probs, prediction, _ = self.model(xs, ilens, ys, tf_rate=tf_rate, sample=False)

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
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_opt, 
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

        judge_scores, unlab_ys_hat, unlab_log_probs = self.sample_and_calculate_judge_probs(unlab_xs, unlab_ilens)
        avg_probs, masked_judge_scores, mask = self.judge.mask_and_average(judge_scores, unlab_ys_hat)
        # baseline: exponential average
        running_average = self.ema(torch.mean(avg_probs))

        # substract baseline
        #judge_scores = (judge_scores - running_average) * mask
        judge_scores = judge_scores * mask

        # pad judge_scores to length of unlab_log_probs
        padded_judge_scores = judge_scores.data.new(judge_scores.size(0), unlab_log_probs.size(1)).fill_(0.)
        padded_judge_scores[:, :judge_scores.size(1)] += judge_scores

        unsup_loss = self.model.mask_and_cal_loss(unlab_log_probs, ys=unlab_ys_hat, mask=padded_judge_scores)

        # mask and calculate loss
        lab_log_probs, _, _ = self.model(lab_xs, lab_ilens, ys=lab_ys, tf_rate=1.0, sample=False)
        sup_loss = self.model.mask_and_cal_loss(lab_log_probs, lab_ys, mask=None)
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

            real_loss = dis_meta['real_loss']
            fake_loss = dis_meta['fake_loss']
            neg_loss = dis_meta['neg_loss']

            acc = dis_meta['acc']

            print(f'Dis:[{d_step + 1}/{d_steps}], '
                    f'real_loss: {real_loss:.3f}, fake_loss: {fake_loss:.3f}, neg_loss: {neg_loss:.3f}'
                    f', acc: {acc:.2f}', end='\r')

            # add to tensorboard
            step = iteration * d_steps + d_step + 1
            tag = self.config['tag']
            for key, val in dis_meta.items():
                self.logger.scalar_summary(f'{tag}/ssl_judge/{key}', val, step)
        print()

        # store model
        model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
        self.save_judge(model_path)


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
                    best_model = self.model.state_dict()
                    print(f'Save #{step} model, val_loss={avg_valid_loss:.3f}, CER={cer:.3f}')
                    print('-----------------')
        
if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
