import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 



class Resblock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(in_channels, in_channels//2, 1), 
                             nn.ReLU(),
                             nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1),
                             nn.ReLU(),
                             nn.Conv2d(in_channels//2, in_channels, 1))
    
    def forward(self, x):
        residual = x
        x = self.main(x)
        return F.relu(x + residual)

class CPCModel(nn.Module): 
    def __init__(self, in_channels, dim, n_resblocks, n_time_steps):
        super().__init__()
        layers = [nn.Conv2d(in_channels, dim, 5, 2, 2), nn.ReLU()]
        for _ in range(n_resblocks):
            layers += [Resblock(dim)]
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)

        self.gru = nn.GRU(dim, 256)
        self.WK = nn.ModuleList([nn.Linear(dim, 256) for _ in range(n_time_steps)])
        self.logsoftmax = nn.LogSoftmax()


    def forward(self, x):
        # x size : batch_size x 9 x 3 x 16 x 16 
        zs = []
        for i in range(x.size(1)):
            zs.append(self.encoder(x[:,i,:,:].squeeze())) # e.i batch_size x 9 x 256
        zs = torch.stack(zs, dim=1).squeeze() # e.i batch_size x 9 x 256
        print('zs size',zs.size())
        t = 4 #torch.randint(1, 8, size=(1,)).long() # 1 to 7, 4
        zt, ztk = zs[:, :t, :], zs[:, t:, :] # 
        print('zt size', zt.size())
        zt = zt.permute(1,0,2)
        out, ct = self.gru(zt) # b x 256 
        preds = []
        
        ct = ct.squeeze()
        print('ct', ct.size())
        for linear in self.WK[t:]:
            preds.append(linear(ct))

        preds = torch.stack(preds, dim=1) # b x 4 x 256
        print('preds', preds.size())
        print('ct', ct.unsqueeze(-1).size())
        ftk = torch.exp(torch.bmm(preds, ct.unsqueeze(-1))).squeeze_() #b x 4 x 1
        sum_ftk = torch.sum(ftk, dim=0) # 1 x 4
        f = self.logsoftmax(ftk/sum_ftk)

        return f




if __name__ == '__main__':
    x = torch.Tensor(size=(2, 9, 3, 16, 16)).zero_()
    cpc = CPCModel(3, 256, 1, 9)

    print(cpc(x))

    





