import time
from threading import Thread

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

    def __init__(self):
        self.run = True

        self.voltage = 0
        self.rc_values = [0] * 6
        self.ena = False
        self.v1 = 0
        self.v2 = 0

        self.ser = serial.Serial("/dev/ttyACM0", 115200)

        self.threads = []
        worker_thread = Thread(target=self.worker)
        worker_thread.start()
        self.threads.append(worker_thread)

        print("Arduino connected.")

    def quit(self):
        self.run = False
        for thread in self.threads:
            thread.join()
        self.ser.close()

    def begin_std_rc(self):
        """
        Begin rc control in another thread.
        """
        rc_thread = Thread(target=standard_rc, args=(self,))
        rc_thread.start()
        self.threads.append(rc_thread)

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


def standard_rc(interface: Interface, interval: float = 0.01):
    """
    Tank steer.
    """
    print("Begin standard rc control.")
    while interface.run:
        interface.ena = interface.rc_values[4] > 0.5
        interface.v1 = interface.rc_values[2] * 2 - 1
        interface.v2 = interface.rc_values[1] * 2 - 1

        time.sleep(interval)
