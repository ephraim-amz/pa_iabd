import ctypes

from utils import computer_plateform, library_mapping


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

train_pmc_model_arg_dict = {
    "model": ctypes.POINTER(PMC),
    "dataset_inputs": ctypes.POINTER(ctypes.c_float),
    "dataset_inputs_size": ctypes.c_size_t,
    "flattened_dataset_outputs": ctypes.POINTER(ctypes.c_float),
    "dataset_outputs_size": ctypes.c_size_t,
    "alpha": ctypes.c_float,
    "epochs": ctypes.c_int32,
    "is_classification": ctypes.c_bool
}

predict_pmc_model_arg_dict = {
    "model": ctypes.POINTER(PMC),
    "sample_inputs": ctypes.POINTER(ctypes.c_float),
    "sample_inputs_size": ctypes.c_size_t,
    "is_classification": ctypes.c_bool
}

save_model_arg_dict = {
    "model": ctypes.POINTER(PMC),
    "filename": ctypes.c_char_p
}

load_model_arg_dict = {
    "path": ctypes.c_char_p
}

get_X_len_arg_dict = {
    "model": ctypes.POINTER(PMC)
}


class PerceptronMC:
    def __init__(self):
        self.lib = ctypes.CDLL(library_mapping.get(computer_plateform))
        self.lib.new_pmc.argtypes = list(new_pmc_model_arg_dict.values())
        self.lib.new_pmc.restype = ctypes.POINTER(PMC)
        self.lib.train_pmc_model.argtypes = list(train_pmc_model_arg_dict.values())
        self.lib.train_pmc_model.restype = None
        self.lib.load_model.argtypes = list(load_model_arg_dict.values())
        self.lib.load_model.restype = ctypes.POINTER(PMC)
        self.lib.save_model.argtypes = list(save_model_arg_dict.values())
        self.lib.save_model.restype = ctypes.c_int
        self.lib.predict_pmc_model.argtypes = list(predict_pmc_model_arg_dict.values())
        self.lib.predict_pmc_model.restype = ctypes.POINTER(ctypes.c_float)
        self.lib.delete_pmc_model.argtypes = [ctypes.POINTER(PMC)]
        self.lib.delete_pmc_model.restype = None
        self.lib.get_X_len.argtypes = list(get_X_len_arg_dict.values())
        self.lib.get_X_len.restype = ctypes.c_int

    def new(self, num_features: ctypes.c_size_t):
        self.lib.new(num_features)

    def train_pmc_model(self, lm, flattened_dataset_inputs, flattened_dataset_expected_outputs, len_input, len_output):
        self.lib.train_regression(lm, flattened_dataset_inputs, flattened_dataset_expected_outputs, len_input,
                                  len_output)

    def train_classification(self, lm, flattened_dataset_inputs, flattened_dataset_expected_outputs, len_input,
                             len_output, lr, epochs):
        self.lib.train_classification(lm, flattened_dataset_inputs, flattened_dataset_expected_outputs, len_input,
                                      len_output, lr, epochs)

    def predict_regression(self, lm, inputs, inputs_size):
        return self.lib.predict_regression(lm, inputs, inputs_size)

    def predict_classification(self, lm, inputs, inputs_size):
        return self.lib.predict_classification(lm, inputs, inputs_size)

    def delete_model(self, pmc_model):
        self.lib.delete_model(pmc_model)

    def save_linear_model(self, pmc_model, filename: bytes):
        is_model_saved = self.lib.save_model(ctypes.byref(pmc_model), filename)
        if not is_model_saved:
            raise IOError("Une erreur est survenue lors de la sauvegarde du modèle")

    def load_linear_model(self, path):
        load_model_result = self.lib.load_pmc_model(path)

        if load_model_result is not None:
            pmc_model_ptr = ctypes.cast(load_model_result, ctypes.POINTER(PMC))
            return pmc_model_ptr
        else:
            raise IOError("Une erreur est survenue lors du chargement du modèle")
