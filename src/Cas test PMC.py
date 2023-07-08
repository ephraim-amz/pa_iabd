import matplotlib.pyplot as plt
import numpy as np
import ctypes
import sys

if __name__ == "__main__":
    computer_plateform = sys.platform
    library_mapping = {
        "linux": r"/home/ephraim/IABD/pa_iabd/src/library/target/debug/liblibrary.so",
        "windows": r"./library/target/debug/liblibrary.dll",
        "darwin": r"./library/target/debug/liblibrary.dylib",
    }

    lib = ctypes.CDLL(library_mapping.get(computer_plateform))


    class PMC(ctypes.Structure):
        _fields_ = [
            ('layers', ctypes.c_size_t),
            ('weights', ctypes.POINTER(ctypes.c_float)),
        ]


    class Veci32(ctypes.Structure):
        _fields_ = [("data", ctypes.POINTER(ctypes.c_int32)),
                    ("length", ctypes.c_size_t),
                    ("capacity", ctypes.c_size_t)]


    class Vecf32(ctypes.Structure):
        _fields_ = [("data", ctypes.POINTER(ctypes.c_float)),
                    ("length", ctypes.c_size_t),
                    ("capacity", ctypes.c_size_t)]


    class Vec2df32(ctypes.Structure):
        _fields_ = [("data", ctypes.POINTER(Vecf32)),
                    ("length", ctypes.c_size_t),
                    ("capacity", ctypes.c_size_t)]


    class Vec3df32(ctypes.Structure):
        _fields_ = [("data", ctypes.POINTER(Vec2df32)),
                    ("length", ctypes.c_size_t),
                    ("capacity", ctypes.c_size_t)]


    new_pmc_model_arg_dict = {
        "neurons_per_layer": ctypes.POINTER(Veci32),
        "layer_size_per_neuron": ctypes.c_size_t,
    }
    lib.new_pmc.argtypes = list(new_pmc_model_arg_dict.values())
    lib.new_pmc.restype = ctypes.POINTER(PMC)

    train_pmc_model_arg_dict = {
        "model": ctypes.POINTER(PMC),
        "dataset_inputs": ctypes.POINTER(ctypes.c_int),
        "lines": ctypes.c_int,
        "columns": ctypes.c_int,
        "dataset_outputs": ctypes.c_float,
        "output_columns": ctypes.c_int32,
        "alpha": ctypes.c_float,
        "nb_iter": ctypes.c_int32,
        "is_classification": ctypes.c_bool
    }
    lib.train_pmc_model.argtypes = list(new_pmc_model_arg_dict.values())
    lib.train_pmc_model.restype = ctypes.POINTER(PMC)

    predict_pmc_model_arg_dict = {
        "model": ctypes.POINTER(PMC),
        "sample_inputs": ctypes.POINTER(ctypes.c_float),
        "columns": ctypes.c_int,
        "is_classification": ctypes.c_bool
    }

    lib.predict_pmc_model.argtypes = list(predict_pmc_model_arg_dict.values())
    lib.predict_pmc_model.restype = ctypes.POINTER(ctypes.c_float)

    lib.delete_pmc_model.argtypes = [ctypes.POINTER(PMC)]
    lib.delete_pmc_model.restype = None