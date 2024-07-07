import argparse
import time

from gen_data import gen_data
from interface import Interface


def rc(args, interface: Interface):
    while True:
        interface.ena = interface.rc_values[4] > 0.5
        interface.v1 = interface.rc_values[2] * 2 - 1
        interface.v2 = interface.rc_values[1] * 2 - 1

        time.sleep(0.03)


def main(interface: Interface):
    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers(dest="command", required=True)

    rc_p = subp.add_parser("rc")

    data_p = subp.add_parser("data")
    data_p.add_argument("--interval", type=float, default=3)

    args = parser.parse_args()

    if args.command == "rc":
        rc(args, interface)
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
