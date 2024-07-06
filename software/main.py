import time
from threading import Thread

import serial


class Interface:
    # Incoming values
    voltage: float
    # 0 to 1 float
    rc_values: list[float]
    # Outgoing values
    ena: bool
    # -1 to 1 float
    v1: float
    v2: float

    def __init__(self):
        self.run = True

        self.voltage = 0
        self.rc_values = [0] * 6
        self.ena = False
        self.v1 = 0
        self.v2 = 0

        self.ser = serial.Serial("/dev/ttyACM0", 115200)

        self.worker_thread = Thread(target=self.worker)
        self.worker_thread.start()

    def quit(self):
        self.run = False
        self.worker_thread.join()
        self.ser.close()

    def worker(self):
        while self.run:
            # Read in
            line = self.ser.readline().decode("utf-8").strip()
            items = line.strip().split(" ")
            if len(items) != 7:
                continue
            self.voltage = float(items[0])
            self.rc_values = [(float(x) - 1000) / 1000 for x in items[1:]]

            # Write out
            v1 = int(self.v1 * 255)
            v2 = int(self.v2 * 255)
            self.ser.write(f"{int(self.ena)} {v1} {v2}\n".encode("utf-8"))


def main(interface: Interface):
    while True:
        print(interface.voltage, interface.rc_values)
        interface.ena = interface.rc_values[4] > 0.5
        interface.v1 = interface.rc_values[2]

        time.sleep(0.5)


if __name__ == "__main__":
    time.sleep(0.1)
    interface = Interface()
    time.sleep(0.1)
    try:
        main(interface)
    finally:
        interface.quit()
