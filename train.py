import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from transform import overlap
from model import CPCModel
from tqdm import tqdm

def compute_acc(logits, labels):
    pred = torch.argmax(logits, dim=1)
    return torch.sum(torch.eq(pred, labels)*1) / labels.size(0)

transform = transforms.Compose([transforms.ToTensor(), overlap(14)])

mnist_dataset = MNIST('data', transform=transform, download=True)
loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

model = CPCModel(1, 64, 3, 9).to('cuda')
opt = torch.optim.Adam(model.parameters())
epochs = 7

l = []
acc = []
for epoch in tqdm(range(epochs)):
    for i, (x, _) in enumerate(loader):

        opt.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            loss, accuracy = model(x.cuda())
            loss.backward()
            opt.step()
        l.append(loss.item())
        acc.append(accuracy)
        if i % 100 == 0 :
            print(f'loss {sum(l[-100:])/100}, accuracy {np.mean(acc[-100:])} at itertion {i} at epoch {epoch}')

model.save_encoder()