import matplotlib.pyplot as plt
import numpy as np
import ctypes
import sys
import os

os.putenv("RUST_BACKTRACE", "full")

if __name__ == "__main__":
    computer_plateform = sys.platform
    library_mapping = {
        "linux": r"./library/target/debug/liblibrary.so",
        "windows": r"./library/target/debug/liblibrary.dll",
        "darwin": r"./library/target/debug/liblibrary.dylib",
    }

    lib = ctypes.CDLL(library_mapping.get(computer_plateform))


    class Veci64(ctypes.Structure):
        _fields_ = [("data", ctypes.POINTER(ctypes.c_int64)),
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


    class PMC(ctypes.Structure):
        _fields_ = [
            ('layers', ctypes.c_uint),
            ('dimensions', ctypes.POINTER(Veci64)),
            ('X', ctypes.POINTER(Vec2df32)),
            ('W', ctypes.POINTER(Vec2df32)),
            ('deltas', ctypes.POINTER(Vec2df32)),
        ]


    new_pmc_model_arg_dict = {
        "dimensions_arr": ctypes.POINTER(ctypes.c_int64),
        "layer_size_per_neuron": ctypes.c_size_t,
    }
    lib.new_pmc.argtypes = list(new_pmc_model_arg_dict.values())
    lib.new_pmc.restype = ctypes.POINTER(PMC)

    train_pmc_model_arg_dict = {
        "model": ctypes.POINTER(PMC),
        "dataset_inputs": ctypes.POINTER(ctypes.c_float),
        "dataset_inputs_size": ctypes.c_size_t,
        "flattened_dataset_outputs": ctypes.POINTER(ctypes.c_float),
        "alpha": ctypes.c_float,
        "epochs": ctypes.c_int32,
        "is_classification": ctypes.c_bool
    }
    lib.train_pmc_model.argtypes = list(train_pmc_model_arg_dict.values())
    lib.train_pmc_model.restype = None

    predict_pmc_model_arg_dict = {
        "model": ctypes.POINTER(PMC),
        "sample_inputs": ctypes.POINTER(ctypes.c_float),
        "sample_inputs_size": ctypes.c_size_t,
        "is_classification": ctypes.c_bool
    }
    lib.predict_pmc_model.argtypes = list(predict_pmc_model_arg_dict.values())
    lib.predict_pmc_model.restype = ctypes.POINTER(ctypes.c_float)

    lib.delete_pmc_model.argtypes = [ctypes.POINTER(PMC)]
    lib.delete_pmc_model.restype = None

    get_X_len_arg_dict = {
        "model": ctypes.POINTER(PMC)
    }
    lib.get_X_len.argtypes = list(get_X_len_arg_dict.values())
    lib.get_X_len.restype = ctypes.c_int

    X = np.array([[1, 1], [2, 3], [3, 3]])
    Y = np.array([1, -1, -1])

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = (ctypes.c_float * len(flattened_outputs))(*flattened_outputs)

    colors = ["blue" if output >= 0 else "red" for output in Y]

    dimensions = [2, 1]
    dimensions_arr = (ctypes.c_int64 * len(dimensions))(*dimensions)

    pmc_model = lib.new_pmc(dimensions_arr, len(dimensions_arr))

    lib.train_pmc_model(
        pmc_model,
        arr_inputs,
        len(flattened_inputs),
        arr_outputs,
        ctypes.c_float(0.001),
        ctypes.c_int32(100),
        ctypes.c_bool(True),
    )

    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(10, 40) for x2 in range(10, 40)]
    sample_inputs = np.array(test_dataset, dtype=np.float32)
    flattened_inputs = sample_inputs.flatten()
    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    predicted_outputs_ptr = lib.predict_pmc_model(
        pmc_model,
        arr_inputs,
        len(flattened_inputs),
        ctypes.c_bool(True),
    )
    predicted_outputs = np.ctypeslib.as_array(
        predicted_outputs_ptr,
        shape=(lib.get_X_len(pmc_model),),
    )

    predicted_outputs_colors = ["blue" if label >= 0 else "red" for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors, s=2)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()

    lib.delete_pmc_model(pmc_model)

    """

    # Linear Multiple

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = (ctypes.c_float * len(flattened_outputs))(*flattened_outputs)

    colors = ["blue" if output >= 0 else "red" for output in Y]

    dimensions = [2, 1]
    dimensions_arr = (ctypes.c_int64 * len(dimensions))(*dimensions)

    pmc_model = lib.new_pmc(dimensions_arr, len(dimensions_arr))

    lib.train_pmc_model(pmc_model, arr_inputs, len(flattened_inputs), arr_outputs, ctypes.c_float(0.001),
                        ctypes.c_int32(100000), ctypes.c_bool(True))

    test_dataset = [[x1 / 9, x2 / 9] for x1 in range(1, 30) for x2 in range(-1, 30)]
    sample_inputs = np.array(test_dataset, dtype=np.float32)
    flattened_inputs = sample_inputs.flatten()

    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    predicted_outputs = []
    for p in range(len(test_dataset)):
        prediction = lib.predict_pmc_model(pmc_model, arr_inputs, len(flattened_inputs), ctypes.c_bool(True))
        arr = np.ctypeslib.as_array(prediction, (lib.get_X_len(pmc_model),))
        predicted_outputs.append(arr[0])

    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()
    lib.delete_pmc_model(pmc_model)

    # XOR


    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = (ctypes.c_float * len(flattened_outputs))(*flattened_outputs)

    colors = ["blue" if output >= 0 else "red" for output in Y]

    dimensions = [2, 2, 1]
    dimensions_arr = (ctypes.c_int64 * len(dimensions))(*dimensions)

    pmc_model = lib.new_pmc(dimensions_arr, len(dimensions_arr))

    lib.train_pmc_model(pmc_model, arr_inputs, len(flattened_inputs), arr_outputs, ctypes.c_float(0.001),
                        ctypes.c_int32(100000), ctypes.c_bool(True))

    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 20) for x2 in range(-10, 20)]
    sample_inputs = np.array(test_dataset, dtype=np.float32)
    flattened_inputs = sample_inputs.flatten()

    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    predicted_outputs = []
    for p in range(len(test_dataset)):
        prediction = lib.predict_pmc_model(pmc_model, arr_inputs, len(flattened_inputs), ctypes.c_bool(True))
        arr = np.ctypeslib.as_array(prediction, (lib.get_X_len(pmc_model),))
        predicted_outputs.append(arr[0])

    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()
    lib.delete_pmc_model(pmc_model)
    
    # Multi Linear 3 Classes
    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in X])

    X = X[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]
    predicted_outputs_colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for
                                output in predicted_outputs]

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = (ctypes.c_float * len(flattened_outputs))(*flattened_outputs)

    colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in
              Y]

    dimensions = [2, 3]
    dimensions_arr = (ctypes.c_int64 * len(dimensions))(*dimensions)

    pmc_model = lib.new_pmc(dimensions_arr, len(dimensions_arr))

    lib.train_pmc_model(pmc_model, arr_inputs, len(flattened_inputs), arr_outputs, ctypes.c_float(0.001),
                        ctypes.c_int32(100000), ctypes.c_bool(True))

    test_dataset = [[x1 / 10, x2 / 10] for x1 in range(-10, 10) for x2 in range(-10, 10)]
    sample_inputs = np.array(test_dataset, dtype=np.float32)
    flattened_inputs = sample_inputs.flatten()

    arr_inputs = (ctypes.c_float * len(flattened_inputs))(*flattened_inputs)

    predicted_outputs = []
    for p in range(len(test_dataset)):
        prediction = lib.predict_pmc_model(pmc_model, arr_inputs, len(flattened_inputs), ctypes.c_bool(True))
        arr = np.ctypeslib.as_array(prediction, (lib.get_X_len(pmc_model),))
        predicted_outputs.append(arr[0])

    predicted_outputs_colors = ['blue' if label >= 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset], [p[1] for p in test_dataset], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()
    lib.delete_pmc_model(pmc_model)
    """
