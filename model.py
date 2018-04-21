import torch 
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils import cc 

def _get_vgg2l_odim(idim, in_channel=1, out_channel=128):
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)
    return int(idim) * out_channel

def _pad_one_frame(inp):
    inp_t = inp.transpose(1, 2)
    out_t = F.pad(inp_t, (0, 1), mode='replicate')
    out = out_t.transpose(1, 2)
    return out

class VGG2L(torch.nn.Module):
    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        self.in_channel = in_channel
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

    def conv_block(self, inp, layers):
        out = inp
        for layer in layers:
            out = F.relu(layer(out))
        out = F.max_pool2d(out, 2, stride=2, ceil_mode=True)
        return out

    def forward(self, xs, ilens):
        # xs = [batch_size, frames, feeature_dim]
        # ilens is a list of frame length of each utterance 
        xs = torch.transpose(
                xs.view(xs.size(0), xs.size(1), self.in_channel, xs.size(2)//self.in_channel), 1, 2)
        xs = self.conv_block(xs, [self.conv1_1, self.conv1_2])
        xs = self.conv_block(xs, [self.conv2_1, self.conv2_2])
        ilens = np.array(np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64) 
        ilens = np.array(np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64).tolist()
        xs = torch.transpose(xs, 1, 2)
        xs = xs.contiguous().view(xs.size(0), xs.size(1), xs.size(2) * xs.size(3))
        return xs, ilens

class pBLSTMLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, subsample, dropout_rate):
        super(pBLSTMLayer, self).__init__()
        self.subsample = subsample
        if subsample > 0:
            self.BLSTM = torch.nn.LSTM(input_dim*2, hidden_dim, 1, bidirectional=True,
                    dropout=dropout_rate, batch_first=True)
        else:
            self.BLSTM = torch.nn.LSTM(input_dim, hidden_dim, 1, bidirectional=True,
                    dropout=dropout_rate, batch_first=True)

    def forward(self, x):
        # x = [batch_size, frames, feature_dim]
        batch_size = x.size(0)
        timesteps = x.size(1)
        input_dim = x.size(2)
        if self.subsample > 0 and timesteps % 2 == 0:
            x = x.contiguous().view(batch_size, timesteps//2, input_dim*2)
        elif self.subsample > 0:
            # pad one frame
            x = _pad_one_frame(x)
            x = x.contiguous().view(batch_size, timesteps//2+1, input_dim*2)
        output, hidden = self.BLSTM(x)
        return output, hidden

class pBLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsamples, dropout_rate):
        super(pBLSTM, self).__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                idim = input_dim
            else:
                # bidirectional
                idim = hidden_dim * 2
            layers.append(pBLSTMLayer(input_dim=idim, hidden_dim=hidden_dim, subsample=subsamples[i], 
                dropout_rate=dropout_rate))
        self.layers = torch.nn.ModuleList(layers)
        self.total_subsample = sum(subsamples)

    def forward(self, x, ilens):
        out = x 
        for i, layer in enumerate(self.layers):
            out, _ = layer(out)
        for i in range(self.total_subsample):
            ilens = np.array(np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64) 
        return out, ilens.tolist()

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsamples, dropout_rate, in_channel=1):
        super(Encoder, self).__init__()
        self.enc1 = VGG2L(in_channel)
        out_channel = _get_vgg2l_odim(input_dim) 
        self.enc2 = pBLSTM(input_dim=out_channel, hidden_dim=hidden_dim, n_layers=n_layers, 
                subsamples=subsamples, dropout_rate=dropout_rate)

    def forward(self, x, ilens):
        out, ilens = self.enc1(x, ilens)
        out, ilens = self.enc2(out, ilens)
        return out, ilens

if __name__ == '__main__':
    net = cc(Encoder(83, 320, 4, [0, 1, 1, 0], dropout_rate=0.3))
    data = cc(Variable(torch.randn(32, 321, 83)))
    ilens = np.ones((32,), dtype=np.int64) * 121
    output, ilens = net(data, ilens)
    print(output.size(), ilens)

