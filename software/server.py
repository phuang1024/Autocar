import argparse
import time
from pathlib import Path
from socket import socket, AF_INET, SOCK_STREAM
from threading import Thread

from conn import *


def get_next_index(dir):
    i = 0
    for f in dir.iterdir():
        if f.is_file() and f.stem.isdigit():
            i = max(i, int(f.stem) + 1)
    return i


def handle_conn(args, conn):
    query = recvobj(conn)

    if query["type"] == "new_data":
        img = query["img"]
        label = query["label"]
        print(f"Received new data: {label}")

        i = get_next_index(args.dir)
        with open(args.dir / f"{i}.jpg", "wb") as f:
            f.write(img)
        with open(args.dir / f"{i}.txt", "w") as f:
            f.write(str(label))

    elif query["type"] == "get_model":
        pass


def socket_main(args, server, run):
    """
    run: List (for mutability) of one element: True or False.
    """
    while run[0]:
        conn, addr = server.accept()
        print(f"Connection from {addr}")

        thread = Thread(target=handle_conn, args=(args, conn))
        thread.start()


def train(args):
    while True:
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7895)
    parser.add_argument("--dir", type=Path, default="server_data", help="Dir with training data.")
    args = parser.parse_args()

    args.dir.mkdir(exist_ok=True, parents=True)

    server = socket(AF_INET, SOCK_STREAM)
    server.bind(("localhost", args.port))
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
