import ctypes
import numpy as np
import os
import sys


class LinearClassifier(ctypes.Structure):
    _fields_ = [
        ('weights', ctypes.POINTER(ctypes.c_float)),
        ('size', ctypes.c_size_t),
    ]


if __name__ == "__main__":
    computer_plateform = sys.platform
    library_mapping = {
        "linux": r"/home/ephraim/IABD/pa_iabd/src/library/target/debug/liblibrary.so",
        "windows": r"./library/target/debug/liblibrary.dll",
        "darwin": r"./library/target/debug/liblibrary.dylib",
    }
    lib = ctypes.CDLL(library_mapping.get(computer_plateform))

    lib.new.argtypes = [ctypes.c_size_t]
    lib.new.restype = ctypes.POINTER(LinearClassifier)
    linear_classifier_object = lib.new(2)

    lib.delete_model.argtypes = [ctypes.POINTER(LinearClassifier)]
    lib.delete_model.restype = None
    lib.delete_model(linear_classifier_object)

    """

    lib.predict.argtypes = [ctypes.POINTER(LinearClassifier),
                            ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_size_t)]
    """
