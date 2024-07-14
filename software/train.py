import argparse
import random
from pathlib import Path

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
        self.aug = torchvision.transforms.Compose([
            T.RandomRotation(5),
            #T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.1),
            T.RandomResizedCrop(256, scale=(0.8, 1.0), antialias=True),
        ])

    def forward(self, x):
        x = self.aug(x)
        return x


class ImageDataset(Dataset):
    rotate_std = 50
    shear_std = 10

    def __init__(self, dir):
        self.dir = dir
        self.indices = []
        self.aug = Augmentation()

        indices = set()
        for file in dir.glob("*.jpg"):
            if (ind := file.name.split(".")[0]).isdigit():
                indices.add(int(ind))
        self.indices = sorted(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        # Read images
        color = torchvision.io.read_image(str(self.dir / f"{i}.rgb.jpg"))
        color = color.float() / 255
        color = torch.mean(color, dim=0, keepdim=True)

        depth = torchvision.io.read_image(str(self.dir / f"{i}.depth.jpg"))
        depth = depth.float() / 255

        depth_conf = torchvision.io.read_image(str(self.dir / f"{i}.depth_conf.jpg"))
        depth_conf = depth_conf.float() / 255

        x = torch.cat([color, depth, depth_conf], dim=0)

        # Read label
        with open(self.dir / f"{i}.txt", "r") as f:
            label = float(f.read())

        if random.random() < 0.3:
            # Apply fake 3D rotation and corresponding label change.
            rotate = int(torch.randn(1).item() * self.rotate_std)
            rotate = min(rotate, 150)
            left = max(rotate, 0)
            width = 256 - abs(rotate)
            x = T.functional.crop(x, top=abs(rotate), left=left, height=width, width=width)
            x = T.functional.resize(x, 256, antialias=True)

            label = label - rotate / self.rotate_std / 3

        """
        elif random.random() < 0.3:
            # Apply fake 3D translation via shear.
            shear = int(torch.randn(1).item() * self.shear_std)
            shear = min(shear, 60)
            x = T.functional.affine(x, angle=0, translate=(0, 0), scale=1.1, shear=shear)
            x = T.functional.resize(x, 256, antialias=True)

            label = label - shear / self.shear_std / 3
        """

        # Apply augmentation
        x = self.aug(x)

        label = torch.clamp(torch.tensor(label).float(), -1, 1)

        return x, label


class AutocarModel(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.resnet = torchvision.models.resnet18()
        self.resnet.fc = torch.nn.Linear(512, 1)
        self.tanh = torch.nn.Tanh()
        self.temperature = temperature

    def forward(self, x):
        x = self.resnet(x)
        x = x * self.temperature
        x = self.tanh(x)
        return x


def preview_data(dataset):
    # Make grid with torchvision
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    x, y = next(iter(loader))
    for i in range(3):
        grid = torchvision.utils.make_grid(x[:, i:i+1, ...], nrow=4)
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
    dataset = ImageDataset(args.dir)
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    loader_args = {
        "batch_size": 16,
        "num_workers": 4,
        "pin_memory": True,
        "shuffle": True,
    }
    train_loader = DataLoader(train_set, **loader_args)
    val_loader = DataLoader(val_set, **loader_args)

    model = AutocarModel().to(DEVICE)
    if args.resume:
        model.load_state_dict(torch.load(args.dir / "model.pt"))
        print("Resumed model from", args.dir / "model.pt")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    gamma = 1e-2 ** (1 / args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    print("LR:", args.lr, "Gamma:", gamma)

    writer = SummaryWriter(args.dir / "logs")
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

        torch.save(model.state_dict(), args.dir / "model.pt")
        scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default="results", help="Dir with training results.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", action="store_true", help="Resume from dir/model.pt")
    args = parser.parse_args()

    args.dir.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
