import argparse
import time

from gen_data import gen_data
from interface import Interface


def main(interface: Interface):
    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers(dest="command", required=True)

    rc_p = subp.add_parser("rc")

    data_p = subp.add_parser("data")
    data_p.add_argument("--interval", type=float, default=3)
    data_p.add_argument("--dir", type=str, required=True)
    data_p.add_argument("--res", type=int, default=256)

    args = parser.parse_args()

    if args.command == "rc":
        interface.begin_std_rc()
        while True:
            print(interface.rc_values)
            time.sleep(0.1)

    elif args.command == "data":
        gen_data(args, interface)


if __name__ == "__main__":
    time.sleep(0.1)
    interface = Interface()
    time.sleep(0.1)
    try:
        main(interface)
    finally:
        interface.quit()

    print("Autocar quit.")
