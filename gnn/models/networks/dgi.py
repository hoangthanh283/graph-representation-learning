import torch
from torch import nn


class DGI(nn.Module):
    def __init__(self, encoder, output_dim):
        super().__init__()
        self.readout = ReadOut()
        self.encoder = encoder
        self.discrimator = Discriminator(input_dim = output_dim)

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, V, A):
        out = self.encoder(V,A)
        return out
    
    def forward_contrastive(self, H_pos, H_neg):
        batch_size = H_pos.shape[0]
        S = self.readout(H_pos)
        
        if not self.discrimator:
            self.discrimator = Discriminator(input_dim = int(S.shape[-1]))

        score_h_pos, score_h_neg = self.discrimator(S, H_pos, H_neg)
        score_h = torch.cat((score_h_pos, score_h_neg),1)
        
        return score_h


class ReadOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigm = nn.Sigmoid()

    def forward(self, V):
        out = torch.mean(V, 1)
        return self.sigm(out)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(input_dim, input_dim, 1)
        self.input_dim = input_dim
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, S, H_pos, H_neg):
        S = torch.unsqueeze(S,1)

        score_pos = torch.squeeze(self.bilinear(H_pos, S.repeat(1, int(H_pos.shape[1]), 1)), 2)
        score_neg = torch.squeeze(self.bilinear(H_neg, S.repeat(1, int(H_neg.shape[1]), 1)), 2)

        return score_pos, score_neg