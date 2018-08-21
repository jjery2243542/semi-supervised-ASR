import torch 

def cc(net):
    if torch.cuda.is_available():
        return net.cuda()
    else:
        return net

def _seq_mask(seq_len, max_len):
    batch_size = seq_len.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    print(seq_range_expand)
    if seq_len.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_len_expand = seq_len.unsqueeze(1).expand_as(seq_range_expand)
    print(seq_len_expand)

    return (seq_range_expand < seq_len_expand).float()
