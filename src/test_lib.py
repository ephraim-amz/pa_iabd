import ctypes
import numpy as np
import os

if __name__ == "__main__":
    lib = ctypes.CDLL(r"./library/target/debug/liblibrary.so")

    lib.add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.add.restype = ctypes.c_int32
    print(lib.add(2, 3))
