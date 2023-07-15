import matplotlib.pyplot as plt
import numpy as np
import ctypes
import sys
import os

os.putenv('RUST_BACKTRACE', 'full')

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

    save_model_arg_dict = {
        "model": ctypes.POINTER(LinearClassifier),
        "filename": ctypes.c_char_p
    }

    lib.save_model.argtypes = list(save_model_arg_dict.values())
    lib.save_model.restype = ctypes.c_int
    filename = b"model.json"
    # is_model_saved = lib.save_model(ctypes.byref(pmc_model), filename)
    # if not is_model_saved:
    #     raise IOError("Une erreur est survenue lors de la sauvegarde du modèle")

    load_model_arg_dict = {
        "path": ctypes.c_char_p
    }

    lib.load_model.argtypes = list(load_model_arg_dict.values())
    lib.load_model.restype = ctypes.POINTER(LinearClassifier)

    # path = filename
    #
    # load_model_result = lib.load_model(path)
    #
    # if load_model_result is not None:
    #     pmc_model_ptr = ctypes.cast(load_model_result, ctypes.POINTER(PMC))
    # else:
    #     raise IOError("Une erreur est survenue lors du chargement du modèle")

    """
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

    test_dataset_inputs = [[x1, x2] for x1 in range(0, 5) for x2 in range(0, 5)]
    colors = ["blue" if output >= 0 else "red" for output in Y]

    lib.train_classification(linear_classifier_object, arr_inputs, arr_outputs, len(flattened_inputs),
                             len(flattened_outputs), 0.31, 100000)

    predicted_outputs = []
    for p in test_dataset_inputs:
        arr_res1 = (ctypes.c_float * len(p))(*p)
        curr = lib.predict_classification(linear_classifier_object, arr_res1, len(p))
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
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0, np.ones((50, 1)) * -1.0])

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    test_dataset_inputs = [[float(x1) / 6, float(x2) / 6] for x1 in range(0, 20) for x2 in
                           range(0, 20)]
    colors = ["blue" if output >= 0 else "red" for output in Y]

    lib.train_classification(linear_classifier_object, arr_inputs, arr_outputs, len(flattened_inputs),
                             len(flattened_outputs),  0.000001, 100000)

    predicted_outputs = []
    for p in test_dataset_inputs:
        arr_res1 = ctypes.c_float * len(p)
        arr_res2 = arr_res1(*p)
        curr = lib.predict_classification(linear_classifier_object, arr_res2, len(p))
        predicted_outputs.append(curr)

    predicted_outputs_colors = ['blue' if label == 1 else 'red' for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset_inputs], [p[1] for p in test_dataset_inputs], c=predicted_outputs_colors)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors[0:100], s=200)
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


    # Cross

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

    linear_classifier_object = lib.new(2)

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


    test_dataset_inputs = [[x1, x2] for x1 in range(0, 2) for x2 in range(0, 2)]
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
    plt.scatter([p[0] for p in test_dataset_inputs], [p[1] for p in test_dataset_inputs], c=predicted_outputs_colors,s=2000)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=20)
    plt.show()
    lib.delete_model(linear_classifier_object)
    

    # Multi Linear 3 classes :

    X = np.random.random((500, 2)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in X])

    X = X[[not np.all(arr == [0, 0, 0]) for arr in X]]
    Y = Y[[not np.all(arr == [0, 0, 0]) for arr in Y]]

    linear_classifier_object = lib.new(3)

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    test_dataset_inputs = [[x1, x2] for x1 in range(-2, 2) for x2 in range(-2, 2)]
    colors = ["blue" if np.argmax(output) == 0 else ("red" if np.argmax(output) == 1 else "green") for output in Y]

    lib.train_classification(linear_classifier_object, arr_inputs, arr_outputs, len(flattened_inputs),
                             len(flattened_outputs), 0.1, 1000)

    predicted_outputs = []
    for p in test_dataset_inputs:
        arr_res1 = ctypes.c_float * len(p)
        arr_res2 = arr_res1(*p)
        curr = lib.predict_classification(linear_classifier_object, arr_res2, len(p))
        predicted_outputs.append(curr)

    predicted_outputs_colors = ["blue" if np.argmax(label) == 0 else ("red" if np.argmax(label) == 1 else "green") for label in predicted_outputs]
    plt.scatter([p[0] for p in test_dataset_inputs], [p[1] for p in test_dataset_inputs], c=predicted_outputs_colors,
                s=2000)
    plt.scatter([p[0] for p in X], [p[1] for p in X], c=colors, s=20)
    plt.show()
    lib.delete_model(linear_classifier_object)
    

    # Multi Cross

    X = np.random.random((1000, 3)) * 2.0 - 1.0
    Y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else
                  [0, 1, 0] if abs(p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else
                  [0, 0, 1] for p in X])

    linear_classifier_object = lib.new(3)

    test_dataset = [[x1, x2] for x1 in range(-2, 2) for x2 in range(-2, 2)]
    predicted_colors = ["blue" if np.argmax(output) == 0
                        else ("red" if np.argmax(output) == 1 else "green") for output in Y]

    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    print(len(flattened_inputs) // X.shape[1], len(flattened_outputs) // Y.shape[1])
    lib.train_classification(linear_classifier_object, arr_inputs, arr_outputs, len(flattened_inputs) // X.shape[1], len(flattened_outputs) // Y.shape[1], 0.1, 1000)


    predicted_outputs = []
    for p in test_dataset:
        arr_res1 = ctypes.c_float * len(p)
        arr_res2 = arr_res1(*p)
        curr = lib.predict_classification(linear_classifier_object, arr_res2, len(p))
        predicted_outputs.append(curr)

    predicted_outputs_colors = ["blue" if np.argmax(label) == 0 else ("red" if np.argmax(label) == 1 else "green") for
                                label in predicted_outputs]

    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][0] == 1, enumerate(X)))))[:, 1],
                color='blue')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][1] == 1, enumerate(X)))))[:, 1],
                color='red')
    plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 0],
                np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]][2] == 1, enumerate(X)))))[:, 1],
                color='green')
    plt.show()
    lib.delete_model(linear_classifier_object)
    """

    # Régression

    X = np.array([
        [1],
        [2]
    ])
    Y = np.array([
        2,
        3
    ])

    test_dataset_inputs = list(map(lambda i: float(i), range(-10, 11)))

    linear_regression_object = lib.new(1)
    flattened_inputs = X.flatten().astype(np.float32)
    arr_inputs = flattened_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    flattened_outputs = Y.flatten().astype(np.float32)
    arr_outputs = flattened_outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.train_regression(linear_regression_object, arr_inputs, arr_outputs, len(flattened_inputs),
                         len(flattened_outputs))

    test_inputs = np.array(test_dataset_inputs, dtype=np.float32, order='C')
    flattened_test_inputs = test_inputs.flatten()
    arr_test_inputs = flattened_test_inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    predicted_outputs = []
    for p in test_dataset_inputs:
        arr_res2 = (ctypes.c_float * 1)(*p)
        print(arr_res2)
        d = lib.predict_regression(linear_regression_object, arr_res2, 1)
        predicted_outputs.append(d)

    plt.plot([p[0] for p in test_dataset_inputs], predicted_outputs)
    plt.scatter([p[0] for p in test_dataset_inputs], predicted_outputs, s=200)
    plt.axis([-10, 10, -10, 10])
    plt.show()
    lib.delete_model(linear_regression_object)
