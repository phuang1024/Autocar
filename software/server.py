import argparse
import time
from pathlib import Path
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split

from conn import *
from model import create_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_next_index(dir):
    i = 0
    for f in dir.iterdir():
        if f.is_file() and f.stem.isdigit():
            i = max(i, int(f.stem) + 1)
    return i


def handle_conn(args, conn):
    query = recvobj(conn)
    results_dir = args.dir / args.name

    if query["type"] == "new_data":
        img = query["img"]
        label = query["label"]
        print(f"Received new data: {label}")

        i = get_next_index(results_dir)
        with open(results_dir / f"{i}.jpg", "wb") as f:
            f.write(img)
        with open(results_dir / f"{i}.txt", "w") as f:
            f.write(str(label))

    elif query["type"] == "get_model":
        print("Sending model...")
        model_path = results_dir / "model.pt"
        if model_path.exists():
            with open(results_dir / "model.pt", "rb") as f:
                model = f.read()
        else:
            model = None
        sendobj(conn, {"model": model})


def socket_main(args, server, run):
    """
    run: List (for mutability) of one element: True or False.
    """
    while run[0]:
        conn, addr = server.accept()
        print(f"Connection from {addr}")

        thread = Thread(target=handle_conn, args=(args, conn))
        thread.start()


class Augmentation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = torchvision.transforms.Compose([
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            T.RandomResizedCrop(256, scale=(0.8, 1.0)),
        ])

    def forward(self, x):
        return self.aug(x)


class ImageDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = list(dir.glob("*.jpg"))
        self.transform = Augmentation()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i].with_suffix(".txt"), "r") as f:
            label = float(f.read())
            label = torch.tensor(label).float()
        img = torchvision.io.read_image(str(self.files[i]))
        img = img.float() / 255
        img = self.transform(img)
        return img, label


def train(args):
    def make_dataloaders():
        dataset = ImageDataset(results_dir)

        train_len = int(len(dataset) * 0.8)
        if train_len == 0:
            return None, None, None
        val_len = len(dataset) - train_len
        if train_len == 0:
            train_len = len(dataset)
            val_len = 0
        train_set, val_set = random_split(dataset, [train_len, val_len])

        loader_args = {
            "batch_size": 32,
            "num_workers": 4,
            "pin_memory": True,
            "shuffle": True,
        }
        train_loader = DataLoader(train_set, **loader_args)
        val_loader = DataLoader(val_set, **loader_args)

        return dataset, train_loader, val_loader

    results_dir = args.dir / args.name

    model = create_model().to(DEVICE)

    dataset = None
    train_loader = None
    val_loader = None

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch = 0
    step = 0
    while True:
        dataset, train_loader, val_loader = make_dataloaders()
        if dataset is None:
            print("No data, waiting...")
            time.sleep(5)
            continue

        #print(f"Train epoch {epoch}, step {step}")
        model.train()
        total_loss = 0
        for img, label in train_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            pred = model(img).squeeze(1)
            loss = criterion(pred, label)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            step += 1
        #print(f"Train loss: {total_loss / len(train_loader)}")

        #print(f"Validation epoch {epoch}")
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for img, label in val_loader:
                img, label = img.to(DEVICE), label.to(DEVICE)
                pred = model(img).squeeze(1)
                loss = criterion(pred, label)
                total_loss += loss.item()
        #print(f"Validation loss: {total_loss / len(val_loader)}")

        torch.save(model.state_dict(), results_dir / "model.pt")

        epoch += 1
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7895)
    parser.add_argument("--dir", type=Path, default="results", help="Dir with training results.")
    parser.add_argument("--name", type=str, default="model", help="Name of the model.")
    args = parser.parse_args()

    results_dir = args.dir / args.name
    results_dir.mkdir(parents=True, exist_ok=True)

    server = socket(AF_INET, SOCK_STREAM)
    server.bind(("0.0.0.0", args.port))
    server.listen()
    print(f"Server is listening on port {args.port}")

    run = [True]
    socket_thread = Thread(target=socket_main, args=(args, server, run))
    socket_thread.start()

    try:
        train(args)
    except KeyboardInterrupt:
        run[0] = False
    finally:
        server.close()

    socket_thread.join()


if __name__ == "__main__":
    main()
