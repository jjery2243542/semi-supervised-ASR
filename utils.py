import torch 
import numpy as np
from tensorboardX import SummaryWriter
import editdistance
import torch.nn as nn
import torch.nn.init as init

def onehot(input_x, encode_dim=None):
    if encode_dim is None:
        encode_dim = torch.max(input_x) + 1
    input_x = input_x.int().unsqueeze(-1)
    return input_x.new_zeros(*input_x.size()[:-1], encode_dim).float().scatter_(-1, input_x, 1)

def sample_gumbel(size, eps=1e-20):
    u = torch.rand(size)
    sample = -torch.log(-torch.log(u + eps) + eps)
    return sample

def gumbel_softmax_sample(logits, temperature=1.):
    y = logits + sample_gumbel(logits.size()).type(logits.type())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1., hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        _, max_ind = torch.max(y, dim=-1, keepdim=True)
        y_hard = torch.zeros_like(y).scatter_(-1, max_ind, 1.0)
        y = (y_hard - y).detach() + y
    return y

class EMA(nn.Module):
    def __init__(self, momentum=0.9):
        super(EMA, self).__init__()
        self.momentum = momentum
        self.last_average = None
        
    def forward(self, x):
        if self.last_average is None:
            new_average = x
        else:
            new_average = (1 - self.momentum) * x + self.momentum * self.last_average
        self.last_average = new_average.detach()
        return new_average
    
    def get_moving_average(self):
        if self.last_average:
            return self.last_average.item()
        else:
            return 0

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
                #init.normal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
                #init.normal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def _inflate(tensor, times, dim):
    repeat_dims = [1] * tensor.dim()
    repeat_dims[dim] = times
    return tensor.repeat(*repeat_dims)

def _inflate_np(np_array, times, dim):
    repeat_dims = [1] * np_array.ndim
    repeat_dims[dim] = times
    return np_array.repeat(repeat_dims)

def adjust_learning_rate(optimizer, lr):
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return lr

def normalize_judge_scores(judge_scores, lengths):
    # judge_scores: [batch_size x frame_length]
    # lengths: [len_1, len_2, ..., len_n]
    for i in range(judge_scores.size(0)):
        miu = torch.mean(judge_scores[i, :lengths[i]])
        std = torch.std(judge_scores[i, :lengths[i]]) + 1e-9
        judge_scores[i, :lengths[i]] = (judge_scores[i, :lengths[i]] - miu) / std
    return

def cc(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return net.to(device)

def to_gpu(data):
    xs, ilens, ys = data
    xs = cc(xs)
    ys = [cc(y) for y in ys]
    return xs, ilens, ys

def to_sents(ind_seq, vocab, non_lang_syms):
    char_list = ind2character(ind_seq, non_lang_syms, vocab) 
    sents = char_list_to_str(char_list)
    return sents

def plot_alignment(alignment, gs, idx, mode):
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}_{}_{}.png'.format(hp.log_dir, mode, idx, gs), format='png')

def pad_list(xs, pad_value=0):
    batch_size = len(xs)
    max_length = max(x.size(0) for x in xs)
    pad = xs[0].data.new(batch_size, max_length, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(batch_size):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def _seq_mask(seq_len, max_len, is_list=True):
    if is_list:
        seq_len = torch.from_numpy(np.array(seq_len))
    batch_size = seq_len.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if seq_len.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_len_expand = seq_len.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_len_expand).float()

def remove_pad_eos(sequences, eos=2):
    sub_sequences = []
    for sequence in sequences:
        try:
            eos_index = next(i for i, v in enumerate(sequence) if v == eos)
        except StopIteration:
            eos_index = len(sequence)
        sub_sequence = sequence[:eos_index]
        sub_sequences.append(sub_sequence)
    return sub_sequences

def remove_pad_eos_batch(sequences, eos=2):
    length = sequences.size(1)
    indices = [length for _ in range(sequences.size(0))]
    for i, elements in enumerate(zip(*sequences)):
        indicators = [element == eos for element in elements]
        indices = [i if index == length and indicator else index for index, indicator in zip(indices, indicators)]
    sub_sequences = [sequence[:index] for index, sequence in zip(indices, sequences)]
    return sub_sequences

def ind2character(sequences, non_lang_syms, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    non_lang_syms_ind = [vocab[sym] for sym in non_lang_syms]
    char_seqs = []
    for sequence in sequences:
        char_seq = [inv_vocab[ind] for ind in sequence if ind not in non_lang_syms_ind]
        #char_seq = [inv_vocab[ind] for ind in sequence]
        char_seqs.append(char_seq)
    return char_seqs

def calculate_cer(hyps, refs):
    total_dis, total_len = 0., 0.
    for hyp, ref in zip(hyps, refs):
        dis = editdistance.eval(hyp, ref)
        total_dis += dis
        total_len += len(ref)
    return total_dis / total_len

def char_list_to_str(char_lists):
    sents = []
    for char_list in char_lists:
        sent = ''.join([char if char != '<space>' else ' ' for char in char_list])
        sents.append(sent)
    return sents

class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
