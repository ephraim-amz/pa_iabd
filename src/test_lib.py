import ctypes
import numpy as np
import os
import sys

if __name__ == "__main__":
    computer_plateform = sys.platform
    library_mapping = {
        "linux": r"./library/target/debug/liblibrary.so",
        "windows": r"./library/target/debug/liblibrary.dll",
        "darwin": r"./library/target/debug/liblibrary.dylib",
    }
    lib = ctypes.CDLL(library_mapping.get(computer_plateform))

    lib.add.argtypes = [ctypes.c_int32, ctypes.c_int32]
    lib.add.restype = ctypes.c_int32
    print(lib.add(2, 3))
