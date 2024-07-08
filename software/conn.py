import pickle
import struct
import time


def sendobj(sock, obj):
    data = pickle.dumps(obj)
    size = struct.pack("<I", len(data))
    sock.sendall(size)
    sock.sendall(data)


def recvobj(sock):
    size = sock.recv(4)
    size = struct.unpack("<I", size)[0]
    data = b""
    i = 0
    while len(data) < size:
        data += sock.recv(size - len(data))
        i += 1
        time.sleep(0.003)
        if i > 10000:
            raise ValueError("recvobj: Too many iterations")
    return pickle.loads(data)
