import time
from threading import Thread

import numpy as np
import serial


class Interface:
    # Incoming values
    voltage: float
    # 0 to 1 float
    # Order: R_h, R_v, L_v, L_h, sw1, sw2
    rc_values: list[float]

    # Outgoing values
    ena: bool
    # -1 to 1 float
    v1: float
    v2: float

    # -1 to 1. Used in auto_rc.
    nn_pred: float

    def __init__(self):
        self.run = True

        self.voltage = 0
        self.rc_values = [0] * 6
        self.ena = False
        self.v1 = 0
        self.v2 = 0
        self.nn_pred = 0

        self.ser = serial.Serial("/dev/ttyACM0", 115200)

        self.threads = []

        self.add_thread(self.worker)

        print("Arduino connected.")

    def quit(self):
        self.run = False
        for thread in self.threads:
            thread.join()
        self.ser.close()

    def add_thread(self, target, args=()):
        thread = Thread(target=target, args=args)
        thread.start()
        self.threads.append(thread)

    def worker(self):
        while self.run:
            # Read in
            try:
                line = self.ser.readline().decode("utf-8").strip()
            except UnicodeDecodeError:
                continue

            items = line.strip().split(" ")
            if len(items) != 7:
                continue
            self.voltage = float(items[0])
            self.rc_values = [(float(x) - 1000) / 1000 for x in items[1:]]

            # Write out
            v1 = int(self.v1 * 255)
            v2 = int(self.v2 * 255)
            self.ser.write(f"{int(self.ena)} {v1} {v2}\n".encode("utf-8"))

    def standard_rc(self):
        """
        Tank steer.
        """
        print("Begin standard rc control.")

        while self.run:
            self.ena = self.rc_values[4] > 0.5
            self.v1 = self.rc_values[2] * 2 - 1
            self.v2 = self.rc_values[1] * 2 - 1

            time.sleep(0.01)

    def auto_rc(self):
        """
        Merges NN's direction prediction with manual RC control/override.

        Update `self.nn_pred` from outside.
        """
        print("Begin auto rc control.")

        while self.run:
            self.ena = self.rc_values[4] > 0.5

            speed = self.rc_values[2]
            steer = self.nn_pred + (self.rc_values[0] * 2 - 1)
            steer *= speed * 1.5
            self.v1 = np.clip(speed + steer, -1, 1)
            self.v2 = np.clip(speed - steer, -1, 1)

            time.sleep(0.01)
