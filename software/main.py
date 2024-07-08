import argparse
import time

from gen_data import gen_data
from interface import Interface
from model import train_main


def main(interface: Interface):
    """
    Possible subcommands:
    rc: Begin standard RC control
    data: Generate data for training
    train: Begin reinforcement learning
    server: Server side, concurrent with `train`
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=256)
    subp = parser.add_subparsers(dest="command", required=True)

    rc_p = subp.add_parser("rc")

    data_p = subp.add_parser("data")
    data_p.add_argument("--interval", type=float, default=3)
    data_p.add_argument("--dir", type=str, required=True)

    train_p = subp.add_parser("train")
    train_p.add_argument("--ip", type=str, default="localhost")
    train_p.add_argument("--port", type=int, default=7895)
    train_p.add_argument("--infer-ival", type=float, default=0.1)
    train_p.add_argument("--new-data-ival", type=float, default=1)
    train_p.add_argument("--new-model-ival", type=float, default=5)

    args = parser.parse_args()

    if args.command == "rc":
        interface.add_thread(interface.standard_rc)
        while True:
            print(interface.rc_values)
            time.sleep(0.1)

    elif args.command == "data":
        gen_data(args, interface)

    elif args.command == "train":
        train_main(args, interface)


if __name__ == "__main__":
    time.sleep(0.1)
    interface = Interface()
    time.sleep(0.1)
    try:
        main(interface)
    finally:
        interface.quit()

    print("Autocar quit.")
