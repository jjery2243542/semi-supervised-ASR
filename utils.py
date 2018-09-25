import torch 
import numpy as np
from tensorboardX import SummaryWriter
import editdistance

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

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

def _seq_mask(seq_len, max_len):
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

def ind2character(sequences, non_lang_syms, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    non_lang_syms_ind = [vocab[sym] for sym in non_lang_syms]
    char_seqs = []
    for sequence in sequences:
        char_seq = [inv_vocab[ind] for ind in sequence if ind not in non_lang_syms_ind]
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
