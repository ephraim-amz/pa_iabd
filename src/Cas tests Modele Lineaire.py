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


    class LinearClassifier(ctypes.Structure):
        _fields_ = [
            ('weights', ctypes.POINTER(ctypes.c_float)),
            ('size', ctypes.c_size_t),
        ]


    # ## Functions initializations

    new_ml_arg_dict = {
        "num_features": ctypes.c_size_t
    }
    lib.new.argtypes = list(new_ml_arg_dict.values())
    lib.new.restype = ctypes.POINTER(LinearClassifier)

    train_regression_arg_dict = {
        'lm': ctypes.POINTER(LinearClassifier),
        'flattened_dataset_inputs': ctypes.POINTER(ctypes.c_float),
        'flattened_dataset_expected_outputs': ctypes.POINTER(ctypes.c_float),
        'len_input': ctypes.c_size_t,
        'len_output': ctypes.c_size_t,
    }
    lib.train_regression.argtypes = list(train_regression_arg_dict.values())
    lib.train_regression.restype = None

    train_classification_arg_dict = {
        'lm': ctypes.POINTER(LinearClassifier),
        'flattened_dataset_inputs': ctypes.POINTER(ctypes.c_float),
        'flattened_dataset_expected_outputs': ctypes.POINTER(ctypes.c_float),
        'len_input': ctypes.c_size_t,
        'len_output': ctypes.c_size_t,
        'lr': ctypes.c_float,
        'epochs': ctypes.c_int
    }
    lib.train_classification.argtypes = list(train_classification_arg_dict.values())
    lib.train_classification.restype = None

    predict_classification_arg_dict = {
        'lm': ctypes.POINTER(LinearClassifier),
        'inputs': ctypes.POINTER(ctypes.c_float),
        'inputs_size': ctypes.c_size_t,
    }
    lib.predict_classification.argtypes = list(predict_classification_arg_dict.values())
    lib.predict_classification.restype = ctypes.c_float

    predict_regression_arg_dict = {
        'lm': ctypes.POINTER(LinearClassifier),
        'inputs': ctypes.POINTER(ctypes.c_float),
        'inputs_size': ctypes.c_size_t,
    }
    lib.predict_regression.argtypes = list(predict_regression_arg_dict.values())
    lib.predict_regression.restype = ctypes.c_float

    lib.delete_model.argtypes = [ctypes.POINTER(LinearClassifier)]
    lib.delete_model.restype = None

    # Cas simple

    linear_classifier_object = lib.new(2)

    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    Y = np.array([
        1,
        -1,
        -1
    ])

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    test_dataset_inputs = [[x1, x2] for x1 in range(-10, 10) for x2 in range(-10, 10)]
    colors = ["blue" if output >= 0 else "red" for output in Y]

    lib.train_classification(linear_classifier_object, arr_inputs, arr_outputs, len(flattened_inputs),
                             len(flattened_outputs), 0.1, 10000)

    predicted_outputs = []
    for p in test_dataset_inputs:
        arr_res1 = ctypes.c_float * len(p)
        arr_res2 = arr_res1(*p)
        curr = lib.predict_classification(linear_classifier_object, arr_res2, len(p))
        predicted_outputs.append(curr)

    predicted_outputs_colors = ['blue' if label == 0 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset_inputs], [p[1] for p in test_dataset_inputs], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()
    lib.delete_model(linear_classifier_object)

    # cas multiple
    linear_classifier_object = lib.new(2)

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    test_dataset_inputs = [[float(x1) / 6, float(x2) / 6] for x1 in range(0, 20) for x2 in
                    range(0, 20)]
    colors = ["blue" if output >= 0 else "red" for output in Y]

    lib.train_classification(linear_classifier_object, arr_inputs, arr_outputs, len(flattened_inputs),
                             len(flattened_outputs), 0.1, 1000)

    predicted_outputs = []
    for p in test_dataset_inputs:
        arr_res1 = ctypes.c_float * len(p)
        arr_res2 = arr_res1(*p)
        curr = lib.predict_classification(linear_classifier_object, arr_res2, len(p))
        predicted_outputs.append(curr)

    predicted_outputs_colors = ['blue' if label == 1 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset_inputs], [p[1] for p in test_dataset_inputs], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()
    lib.delete_model(linear_classifier_object)

    # XOR

    linear_classifier_object = lib.new(2)

    X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    test_dataset_inputs = [[x1 / 15, x2 / 15] for x1 in range(0, 20) for x2 in range(0, 20)]
    colors = ["blue" if output >= 0 else "red" for output in Y]

    lib.train_classification(linear_classifier_object, arr_inputs, arr_outputs, len(flattened_inputs),
                             len(flattened_outputs), 0.1, 1000)

    predicted_outputs = []
    for p in test_dataset_inputs:
        arr_res1 = ctypes.c_float * len(p)
        arr_res2 = arr_res1(*p)
        curr = lib.predict_classification(linear_classifier_object, arr_res2, len(p))
        predicted_outputs.append(curr)

    predicted_outputs_colors = ['blue' if label == 1 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset_inputs], [p[1] for p in test_dataset_inputs], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=200)
    plt.show()
    lib.delete_model(linear_classifier_object)