import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 



class CPCModel(nn.Module): 
    def __init__(self, in_channels, dim, n_resblocks, n_time_steps):
        layers = [nn.Conv2d(in_channels, dim, 5, 2, 2), nn.ReLU()]
        for _ in range(n_resblocks):
            layers += [Resblock(dim)]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)

        self.gru = nn.GRU(dim, 256)
        self.WK = nn.ModuleList([nn.Linear(dim, 256) for _ in range(n_time_steps)])
        self.logsoftmax = nn.LogSoftmax()


    def forward(self, x, samples):
        # x size : batch_size x 9 x 3 x 16 x 16 
        zs = []
        for _ in range(x.size(1)):
            zs.append(self.encoder(x)) # e.i batch_size x 9 x dim
        zs = torch.stack(zs, dim=1)
        t = torch.randint(1, 8, size=(1,)).long() # 1 to 7, 4
        zt, ztk = zs[:, :t, :], zs[:, t:, :]

        out, ct = self.gru(zt) # b x 256 
        linear = self.WK[t:]
        preds = []
        for _ in range(9-t):
            preds.append(linear(ct))

        preds = torch.stack(preds, dim=1) # b x 4 x 256

        fk = torch.bmm()

        
    







class Resblock(nn.Module):
    def __init__(self, in_channels):
        self.main = nn.Sequential(nn.Conv2d(in_channels, in_channels//2, 1), 
                             nn.ReLU(),
                             nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels//2, in_channels, 1))
    
    def forward(self, x):
        residual = x
        x = self.main(x)
        return F.relu(x + residual)