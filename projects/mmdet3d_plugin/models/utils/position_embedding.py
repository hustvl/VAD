import torch
import torch.nn as nn
import math

class RelPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, pos_norm=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.fc = nn.Linear(4, self.num_pos_feats,bias=False)
        #nn.init.orthogonal_(self.fc.weight)
        #self.fc.weight.requires_grad = False
        self.pos_norm = pos_norm
        if self.pos_norm:
            self.norm = nn.LayerNorm(self.num_pos_feats)
    def forward(self, tensor):
        #mask = nesttensor.mask
        B,C,H,W = tensor.shape
        #print('tensor.shape',  tensor.shape)
        y_range = (torch.arange(H) / float(H - 1)).to(tensor.device)
        #y_axis = torch.stack((y_range, 1-y_range),dim=1)
        y_axis = torch.stack((torch.cos(y_range * math.pi), torch.sin(y_range * math.pi)), dim=1)
        y_axis = y_axis.reshape(H, 1, 2).repeat(1, W, 1).reshape(H * W, 2)

        x_range = (torch.arange(W) / float(W - 1)).to(tensor.device)
        #x_axis =torch.stack((x_range,1-x_range),dim=1)
        x_axis = torch.stack((torch.cos(x_range * math.pi), torch.sin(x_range * math.pi)), dim=1)
        x_axis = x_axis.reshape(1, W, 2).repeat(H, 1, 1).reshape(H * W, 2)
        x_pos = torch.cat((y_axis, x_axis), dim=1)
        x_pos = self.fc(x_pos)

        if self.pos_norm:
            x_pos = self.norm(x_pos)
        #print('xpos,', x_pos.max(),x_pos.min())
        return x_pos


class SineEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels
        """
        super(SineEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (sin(2^k x), cos(2^k x), ...) 
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


# if __name__ == '__main__':
#     pe = Embedding(in_channels=2, N_freqs=64)
#     x_pe = pe(torch.randn(1, 4, 2))
#     a = 0