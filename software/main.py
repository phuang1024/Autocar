import argparse
import time
from pathlib import Path

from gen_data import gen_data_main
from interface import Interface
from auto import auto_main


def main(interface: Interface):
    """
    Possible subcommands:
    rc: Begin standard RC control
    data: Generate data for training
    auto: Auto driving with optional intervention
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=int, default=256)
    subp = parser.add_subparsers(dest="command", required=True)

    rc_p = subp.add_parser("rc")

    data_p = subp.add_parser("data")
    data_p.add_argument("--self-rc", action="store_true", help="Drive auto avoiding obstacles while getting data.")
    data_p.add_argument("--interval", type=float, default=0)
    data_p.add_argument("--dir", type=str, required=True)

    auto_p = subp.add_parser("auto")
    auto_p.add_argument("--model-path", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "rc":
        interface.add_thread(interface.standard_rc)
        while True:
            print(interface.rc_values)
            time.sleep(0.1)

    elif args.command == "data":
        gen_data_main(args, interface)

    elif args.command == "auto":
        auto_main(args, interface)


if __name__ == "__main__":
    time.sleep(0.1)
    interface = Interface()
    time.sleep(0.1)
    try:
        main(interface)
    finally:
        interface.quit()

    print("Autocar quit.")
