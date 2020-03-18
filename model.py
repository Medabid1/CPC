import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

from itertools import combinations_with_replacement


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
        self.WK = nn.ModuleList([nn.Linear(256, dim) for _ in range(n_time_steps)])
        self.logsoftmax = nn.LogSoftmax(dim=0)


    def forward(self, x):
        # x size : batch_size x 9 x 3 x 16 x 16 
        b = x.size(0)
        zs = []
        for i in range(x.size(1)):
            zs.append(self.encoder(x[:,i,:, :,:])) # e.i batch_size x 9 x 256
        zs = torch.stack(zs, dim=1).squeeze() # e.i batch_size x 9 x 256
        
        t = 4 #torch.randint(1, 8, size=(1,)).long() # 1 to 7, 4
        zt, ztk = zs[:, :t, :], zs[:, t:, :] # 
        zt = zt.permute(1,0,2)
        out, ct = self.gru(zt) # b x 256 
        preds = []
        
        ct = ct.squeeze()
        
        for linear in self.WK[t:]:
            preds.append(linear(ct))
        preds = torch.stack(preds, dim=1) # b x 4 x 256
        total_loss = []
        accuracies = []
        for i in range(b):
            p_b = preds.select(0,i)
            ftk = self.logsoftmax(torch.exp(torch.sum(torch.mul(ztk, p_b), dim=-1)).squeeze())
            m = torch.argmax(ftk, dim=0).detach()
            
            t = torch.tensor([i] * 5).cuda()
            
            acc = torch.sum(torch.eq(m,t))/5
            total_loss.append(ftk)
            accuracies.append(acc.cpu().item())

        total_loss = torch.stack(total_loss, dim=0)
        targets = [torch.ones(size=(5,)).long().cuda() * i for i in range(b)]
        targets = torch.stack(targets,dim=0)
        
        total_loss = F.nll_loss(total_loss, targets)
        return total_loss, np.mean(accuracies)




if __name__ == '__main__':
    x = torch.Tensor(size=(32, 9, 1, 14, 14)).zero_()
    cpc = CPCModel(1, 64, 1, 9)

    print(cpc(x))

    





