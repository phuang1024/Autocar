import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Augmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rot = T.RandomRotation(3)
        self.crop = T.RandomResizedCrop(256, scale=(0.8, 1.0), antialias=True)
        self.color = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2)

        self.upper_mask = torch.zeros(256, 256)
        for i in range(192):
            self.upper_mask[i] = 1 - i / 192

    def upper_noise(self, x):
        return x + torch.randn_like(x) * self.upper_mask

    def forward(self, x):
        if random.random() < 0.5:
            x = self.rot(x)
        if random.random() < 0.2:
            x = self.crop(x)
        if random.random() < 0.5:
            x[..., :3, :, :] = self.color(x[..., :3, :, :])
        if random.random() < 0.2:
            x = self.upper_noise(x)
        return x


class ImageDataset(Dataset):
    ts_offset = 5
    """Time series offset. Positive means future label associated with image."""

    rotate_std = 50
    shear_std = 10

    def __init__(self, dir: Path):
        self.dir = dir
        self.indices = []
        self.aug = Augmentation()

        indices = set()
        for file in dir.glob("*.pt"):
            if file.stem.isdigit():
                indices.add(int(file.stem))
        self.indices = sorted(indices)

    def __len__(self):
        return len(self.indices) - self.ts_offset

    def __getitem__(self, i):
        i_x = self.indices[i]
        x = torch.load(self.dir / f"{i_x}.pt")
        x = x.float() / 255

        i_y = self.indices[i + self.ts_offset]
        with open(self.dir / f"{i_y}.txt", "r") as f:
            label = float(f.read())

        x, label = self.aug_3dtrans(x, label)
        x = self.aug(x)

        label = torch.clamp(torch.tensor(label).float(), -1, 1)

        return x, label

    def aug_3dtrans(self, x, label):
        """
        Simulated 3D transform augmentations:
        - Rotation: Simulate 3D Z rotation via horizontal crop.
        - X translation: Simulate 3D horizontal translation via shear.
        - Z translation: Simulate 3D front/back translation via zoom in and depth adjust.
        """
        # Simulated 3D transform augmentations
        if random.random() < 0.3:
            # Rotation augmentation
            rotate = int(torch.randn(1).item() * self.rotate_std)
            rotate = min(rotate, 150)
            left = max(rotate, 0)
            width = 256 - abs(rotate)
            x = T.functional.crop(x, top=abs(rotate), left=left, height=width, width=width)

            label = label - rotate / self.rotate_std / 3

        elif random.random() < 0.3:
            # Z translation augmentation
            zoom = random.uniform(0, 1)
            zoom_px = int(zoom * 50)
            x = x[:, zoom_px:256-zoom_px, zoom_px:256-zoom_px]

            fac = np.interp(zoom, [0, 1], [1, 2])

            # Adjust depth channel.
            x[3] = torch.clamp(x[3] * fac, 0, 1)

            label = label * fac

        """
        elif random.random() < 0.3:
            # X translation augmentation
            shear = int(torch.randn(1).item() * self.shear_std)
            shear = min(shear, 60)
            x = T.functional.affine(x, angle=0, translate=(0, 0), scale=1.1, shear=shear)

            label = label - shear / self.shear_std / 3
        """

        x = T.functional.resize(x, 256, antialias=True)

        return x, label


class AutocarModel(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = torch.nn.Linear(512, 1)

        self.tanh = torch.nn.Tanh()
        self.temperature = temperature

    def forward(self, x):
        x = self.resnet(x)
        x = x * self.temperature
        x = self.tanh(x)
        return x


class OnnxAutocarModel(AutocarModel):
    """
    Inputs are raw depthai data.
    """

    def forward(self, rgb, depth):
        rgb = rgb.float() / 255
        depth = depth.float() / 255
        x = torch.cat([rgb, depth], dim=1)
        return super().forward(x)


def preview_data(dataset):
    # Make grid with torchvision
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    x, y = next(iter(loader))

    grid = torchvision.utils.make_grid(x[:, 0:3, ...], nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

    grid = torchvision.utils.make_grid(x[:, 3:4, ...], nrow=4)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def histogram(model, dataset):
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

    model.eval()
    with torch.no_grad():
        preds = []
        for x, _ in tqdm(loader, desc="Histogram"):
            x = x.to(DEVICE)
            y = model(x).squeeze(1)
            preds.extend(y.cpu().numpy())
        plt.hist(preds, bins=40)
        plt.show()


def train(args):
    dataset = ImageDataset(args.data)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    loader_args = {
        "batch_size": 16,
        "num_workers": 16,
        "pin_memory": True,
        "shuffle": True,
    }
    train_loader = DataLoader(train_set, **loader_args)
    val_loader = DataLoader(val_set, **loader_args)

    model = AutocarModel().to(DEVICE)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
        print("Resumed model from", args.resume)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    gamma = 1e-1 ** (1 / args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    print("LR:", args.lr, "Gamma:", gamma)

    writer = SummaryWriter(args.results / "logs")
    step = 0

    #preview_data(dataset)
    #stop
    #histogram(model, dataset)
    #stop

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in (pbar := tqdm(train_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x).squeeze(1)
            loss = criterion(pred, y)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            step += 1

            pbar.set_description(f"Train: Epoch {epoch + 1}/{args.epochs}, loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), step)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for x, y in (pbar := tqdm(val_loader)):
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).squeeze(1)
                loss = criterion(pred, y)
                total_loss += loss.item()

                pbar.set_description(f"Test: Epoch {epoch + 1}/{args.epochs}, loss: {loss.item():.4f}")
            writer.add_scalar("val_loss", total_loss / len(val_loader), epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), args.results / "model.pt")
        scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="Data directory.")
    parser.add_argument("--results", type=Path, help="Results directory.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", help="Resume from given model path.")
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
