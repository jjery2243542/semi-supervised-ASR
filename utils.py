import torch 

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

def pad_list(xs, pad_value=0):
    batch_size = len(xs)
    max_length = max(x.size(0) for x in xs)
    pad = xs[0].data.new(batch_size, max_length, *xs[0].size()[1:]).zero_() + pad_value
    for i in range(batch_size):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad

def _seq_mask(seq_len, max_len):
    batch_size = seq_len.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if seq_len.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_len_expand = seq_len.unsqueeze(1).expand_as(seq_range_expand)
    return (seq_range_expand < seq_len_expand).float()
