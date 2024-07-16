import argparse
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
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
            T.RandomResizedCrop(256, scale=(0.8, 1.0), antialias=True),
        ])

    def forward(self, x):
        x = self.aug(x)
        return x


class ImageDataset(Dataset):
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
        return len(self.indices)

    def __getitem__(self, i):
        # Read images
        x = torch.load(self.dir / f"{i}.pt")
        x = x.float() / 255

        # Read label
        with open(self.dir / f"{i}.txt", "r") as f:
            label = float(f.read())

        x, label = sim_3d_trans(x, label)
        x = self.aug(x)

        label = torch.clamp(torch.tensor(label).float(), -1, 1)

        return x, label


def sim_3d_trans(x, label, rotate_std=50, shear_std=10):
    """
    Simulated 3D transform augmentations:
    - Rotation: Simulate 3D Z rotation via horizontal crop.
    - X translation: Simulate 3D horizontal translation via shear.
    - Z translation: Simulate 3D front/back translation via zoom in and depth adjust.
    """
    # Simulated 3D transform augmentations
    if random.random() < 0.3:
        # Rotation augmentation
        rotate = int(torch.randn(1).item() * rotate_std)
        rotate = max(min(rotate, 130), -130)
        left = max(rotate, 0)
        width = 256 - abs(rotate)
        x = T.functional.crop(x, top=abs(rotate), left=left, height=width, width=width)

        label = label - rotate / rotate_std / 3

    elif random.random() < 0.3:
        # Z translation augmentation
        zoom = random.uniform(0, 1)
        zoom_px = int(zoom * 50)
        x = x[..., :, zoom_px:256-zoom_px, zoom_px:256-zoom_px]

        fac = np.interp(zoom, [0, 1], [1, 2])

        # Adjust depth channel.
        x[..., 3, :, :] = torch.clamp(x[..., 3, :, :] * fac, 0, 1)

        label = label * fac

    """
    elif random.random() < 0.3:
        # X translation augmentation
        shear = int(torch.randn(1).item() * shear_std)
        shear = min(shear, 60)
        x = T.functional.affine(x, angle=0, translate=(0, 0), scale=1.1, shear=shear)

        label = label - shear / shear_std / 3
    """

    x = T.functional.resize(x, 256, antialias=True)

    return x, label


class AutocarModel(torch.nn.Module):
    em_size = 512

    def __init__(self, temperature=0.1):
        super().__init__()

        self.resnet = torchvision.models.resnet18()
        self.resnet.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = torch.nn.Linear(512, self.em_size)

        self.fc = torch.nn.Linear(self.em_size, 1)
        self.tanh = torch.nn.Tanh()
        self.temperature = temperature

    def forward(self, x):
        em = self.resnet(x)
        x = self.fc(em)
        x = x * self.temperature
        x = self.tanh(x)
        return x, em


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
        "num_workers": 4,
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
    gamma = 1e-2 ** (1 / args.epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    print("LR:", args.lr, "Gamma:", gamma)

    writer = SummaryWriter(args.results / "logs")
    step = 0

    # Utils for contrastive learning
    color_jitter = T.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.1)
    upper_mask = torch.ones(256, 256, device=DEVICE)
    for i in range(128):
        upper_mask[i] = 1 - i / 128
    upper_mask = torch.clamp(upper_mask, 0, 1)
    upper_mask = upper_mask.view(1, 1, 256, 256)

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
            pred, _ = model(x)
            loss = criterion(pred.squeeze(1), y)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            step += 1

            pbar.set_description(f"E{epoch + 1}/{args.epochs}: Train: loss={loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), step)

        if (epoch + 1) % args.contr_interval == 0:
            total_loss = 0
            for x, _ in (pbar := tqdm(train_loader)):
                x = x.to(DEVICE)

                # Color jitter
                optimizer.zero_grad()
                with torch.no_grad():
                    x2 = x.clone()
                    x2[:, :3, ...] = color_jitter(x[:, :3, ...])
                _, pred1 = model(x)
                _, pred2 = model(x2)
                loss = F.cosine_similarity(pred1, pred2, dim=-1).mean()
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

                # Upper mask noise
                optimizer.zero_grad()
                noise = torch.randn_like(x, device=DEVICE) * upper_mask
                with torch.no_grad():
                    x2 = x + noise
                _, pred1 = model(x)
                _, pred2 = model(x2)
                loss = F.cosine_similarity(pred1, pred2, dim=-1).mean()
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

                # Simulated 3D transform
                optimizer.zero_grad()
                with torch.no_grad():
                    x2 = sim_3d_trans(x, 0)[0]
                _, pred1 = model(x)
                _, pred2 = model(x2)
                loss = F.cosine_similarity(pred1, pred2, dim=-1).mean() * -1
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

                pbar.set_description(f"E{epoch + 1}/{args.epochs}: Contrastive")
            writer.add_scalar("contr_loss", total_loss / len(train_loader), step)


        model.eval()
        with torch.no_grad():
            total_loss = 0
            for x, y in (pbar := tqdm(val_loader)):
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred, _ = model(x)
                loss = criterion(pred.squeeze(1), y)
                total_loss += loss.item()

                pbar.set_description(f"E{epoch + 1}/{args.epochs}: Train: loss={loss.item():.4f}")
            writer.add_scalar("val_loss", total_loss / len(val_loader), epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), args.results / "model.pt")
        scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="Data directory.")
    parser.add_argument("--results", type=Path, help="Results directory.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--contr-interval", type=int, default=8, help="Contrastive epoch every N epochs.")
    parser.add_argument("--resume", help="Resume from given model path.")
    args = parser.parse_args()

    args.results.mkdir(parents=True, exist_ok=True)

    train(args)


if __name__ == "__main__":
    main()
