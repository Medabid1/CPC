import torch 

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from dataset import overlap
from model import CPCModel

transform = transforms.Compose([transforms.ToTensor(), overlap(14),  transforms.Normalize((0), (0.5))])

mnist_dataset = MNIST('data', transform=transform, download=True)
loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

model = CPCModel(1, 64, 3, 9)
opt = torch.optim.Adam(model.parameters())
epochs = 15 

l = []
for epoch in range(epochs):
    for i, (x, _) in enumerate(loader):
        opt.zero_grad()
        loss = model(x)
        loss.backward()
        opt.step()
        l.append(loss.item())
    if i +1 % 100 == 0 :
        print(f'loss {sum(l[:-100])/100} at itertion {i} at epoch {epoch})
