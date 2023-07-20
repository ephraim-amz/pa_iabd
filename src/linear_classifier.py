import ctypes
from flask_app.utils import library_mapping, computer_plateform


class LinearClassifier(ctypes.Structure):
    _fields_ = [
        ('weights', ctypes.POINTER(ctypes.c_float)),
        ('size', ctypes.c_size_t),
    ]


new_ml_arg_dict = {
    "num_features": ctypes.c_size_t
}

train_regression_arg_dict = {
    'lm': ctypes.POINTER(LinearClassifier),
    'flattened_dataset_inputs': ctypes.POINTER(ctypes.c_float),
    'flattened_dataset_expected_outputs': ctypes.POINTER(ctypes.c_float),
    'len_input': ctypes.c_size_t,
    'len_output': ctypes.c_size_t,
}

train_classification_arg_dict = {
    'lm': ctypes.POINTER(LinearClassifier),
    'flattened_dataset_inputs': ctypes.POINTER(ctypes.c_float),
    'flattened_dataset_expected_outputs': ctypes.POINTER(ctypes.c_float),
    'len_input': ctypes.c_size_t,
    'len_output': ctypes.c_size_t,
    'lr': ctypes.c_float,
    'epochs': ctypes.c_int
}

predict_classification_arg_dict = {
    'lm': ctypes.POINTER(LinearClassifier),
    'inputs': ctypes.POINTER(ctypes.c_float),
    'inputs_size': ctypes.c_size_t,
}

predict_regression_arg_dict = {
    'lm': ctypes.POINTER(LinearClassifier),
    'inputs': ctypes.POINTER(ctypes.c_float),
    'inputs_size': ctypes.c_size_t,
}

save_model_arg_dict = {
    "model": ctypes.POINTER(LinearClassifier),
    "filename": ctypes.c_char_p
}

load_model_arg_dict = {
    "path": ctypes.c_char_p
}


class LinearClassifierModel:
    def __init__(self):
        self.lib = ctypes.CDLL(library_mapping.get(computer_plateform))
        self.lib.new.argtypes = list(new_ml_arg_dict.values())
        self.lib.new.restype = ctypes.POINTER(LinearClassifier)
        self.lib.train_regression.argtypes = list(train_regression_arg_dict.values())
        self.lib.train_regression.restype = None
        self.lib.save_linear_model.argtypes = list(save_model_arg_dict.values())
        self.lib.save_linear_model.restype = ctypes.c_int
        self.lib.load_linear_model.argtypes = list(load_model_arg_dict.values())
        self.lib.load_linear_model.restype = ctypes.POINTER(LinearClassifier)
        self.lib.predict_regression.argtypes = list(predict_regression_arg_dict.values())
        self.lib.predict_regression.restype = ctypes.c_float
        self.lib.delete_linear_model.argtypes = [ctypes.POINTER(LinearClassifier)]
        self.lib.delete_linear_model.restype = None
        self.lib.predict_classification.argtypes = list(predict_classification_arg_dict.values())
        self.lib.predict_classification.restype = ctypes.c_float
        self.lib.train_classification.argtypes = list(train_classification_arg_dict.values())
        self.lib.train_classification.restype = None

    def new(self, num_features: ctypes.c_size_t):
        self.lib.new(num_features)

    def train_regression(self, lm, flattened_dataset_inputs, flattened_dataset_expected_outputs, len_input, len_output):
        self.lib.train_regression(lm, flattened_dataset_inputs, flattened_dataset_expected_outputs, len_input,
                                  len_output)

    def train_classification(
            self,
            lm,
            flattened_dataset_inputs,
            flattened_dataset_expected_outputs,
            len_input,
            len_output,
            lr,
            epochs):
        self.lib.train_classification(lm, flattened_dataset_inputs, flattened_dataset_expected_outputs, len_input,
                                      len_output, lr, epochs)

    def predict_regression(self, lm, inputs, inputs_size):
        return self.lib.predict_regression(lm, inputs, inputs_size)

    def predict_classification(self, lm, inputs, inputs_size):
        return self.lib.predict_classification(lm, inputs, inputs_size)

    def delete_linear_model(self, pmc_model):
        self.lib.delete_linear_model(pmc_model)

    def save_linear_model(self, linear_model, filename: str):
        is_model_saved = self.lib.save_linear_model(ctypes.byref(linear_model), filename.encode('utf-8'))
        if not is_model_saved:
            raise IOError("Une erreur est survenue lors de la sauvegarde du modèle")

    def load_linear_model(self, path: str):
        load_model_result = self.lib.load_linear_model(path.encode('utf-8'))

        if load_model_result is not None:
            linear_model_ptr = ctypes.cast(load_model_result, ctypes.POINTER(LinearClassifier))
            return linear_model_ptr
        else:
            raise IOError("Une erreur est survenue lors du chargement du modèle")
