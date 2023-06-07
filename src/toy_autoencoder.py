import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import sys
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_size = 91
img_transform = transforms.Compose([
    transforms.Normalize([0.5], [0.5])
])


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), stride=(3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(16, 32, (3, 3), stride=(3, 3), padding=(1, 1)), # 32  10
            nn.ReLU(True),
            nn.Conv2d(32, 64, (3, 3), stride=(2, 2), padding=(1, 1)),  # 8 3
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, (3, 3), stride=(3, 3), padding=(1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, (3, 3), stride=(3, 3), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class Stl10Dataset(Dataset):
    #https://github.com/mttk/STL10/blob/master/stl10_input.py
    def __init__(self, path, files):
        self.path = path
        self.files = files

    def __getitem__(self, item):
        file = self.files[item]
        img = cv2.imread(os.path.join(self.path, file), cv2.IMREAD_GRAYSCALE) / 255.
        img = cv2.resize(img, (model_size, model_size))
        img = torch.tensor(img).float()
        label = int(file.split(os.sep)[0]) - 1
        label = torch.tensor(label)

        img = img.to(device)
        label = label.to(device)

        return img, label

    def __len__(self):
        return len(self.files)


def train_batch(input, model, loss_func, optimizer):
    model.train()
    optimizer.zero_grad()
    res = model(input)
    loss = loss_func(res, input)
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss


def split_files(path, divs):
    if divs[0] + divs[1] + divs[2] != 1.0:
        print(f'Wrong divisions: Train={divs[0]} Validation={divs[1]} Test={divs[2]} Total={divs[0] + divs[1] + divs[2]}')
        sys.exit()

    dirs = np.array(os.listdir(path))
    trains = []
    tests = []
    vals = []
    for dir in dirs:
        files = np.array(os.listdir(os.path.join(path, dir)))
        size = len(files)
        np.random.shuffle(files)
        train = int(size * divs[0])
        test = int(size * divs[2])
        val = int(size * divs[1])
        total = train + test + val
        train = train + (size - total)
        prefs = [os.path.join(dir, file) for file in files]
        trains += prefs[:train]
        vals += prefs[train:train + val]
        tests += prefs[train + val:]

    return trains, vals, tests


@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()


def main():
    results_path = '../results/transposed_conv'
    os.makedirs(results_path, exist_ok=True)
    dataset = 'Stl10'
    data_path = 'your path'
    l_rate = 0.001
    train_files, val_files, _ = split_files(data_path, (0.7, 0.3, 0.0))
    train_dset = Stl10Dataset(data_path, train_files)
    val_dset = Stl10Dataset(data_path, val_files)
    train_dld = DataLoader(train_dset, shuffle=True)
    val_dld = DataLoader(val_dset, shuffle=False)

    model = ConvAutoEncoder().to(device)
    summary(model, input_size=(1, model_size, model_size))

    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=l_rate, weight_decay=1e-5)
    num_epochs = 51
    loss_train_epochs = []
    loss_val_epochs = []
    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        loss_batch = []
        for ix, (data, _) in enumerate(train_dld):
            loss = train_batch(data, model, loss_func, optimizer)
            loss_batch.append(loss.detach().cpu().numpy())
        loss_train_epochs.append(np.mean((loss_batch)))

        loss_batch = []
        for ix, (data, _) in enumerate(val_dld):
            loss = validate_batch(data, model, loss_func)
            loss_batch.append(loss.detach().cpu().numpy())
        loss_val_epochs.append(np.mean((loss_batch)))

        pbar.update(1)

    torch.save(model, os.path.join(results_path, f'auto_encoder_{num_epochs}epochs_{l_rate}lr_{dataset}_DS.pth'))

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(loss_val_epochs,label="val")
    plt.plot(loss_train_epochs,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig(os.path.join(results_path, f'metrics_unet_docs_{num_epochs}epochs_{l_rate}lr_vgg16_{dataset}_DS.png'))
    plt.show()


if __name__ == '__main__':
    main()

