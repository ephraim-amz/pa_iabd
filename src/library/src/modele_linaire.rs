use rand::Rng;
use nalgebra::{DMatrix, DVector};


#[repr(C)]
pub struct LinearClassifier {
    pub weights: *mut f32,
    pub size: usize,
}


#[no_mangle]
pub extern "C" fn new(num_features: usize) -> *mut LinearClassifier {
    let mut rng = rand::thread_rng();
    let model = Box::new(LinearClassifier {
        size: num_features,
        weights: Box::into_raw(vec![rng.gen_range(-1.0..1.0); num_features].into_boxed_slice()) as *mut f32,
    });
    let leak_lm = Box::leak(model);
    leak_lm
}


#[no_mangle]
pub extern "C" fn train_regression(
    lm: *mut LinearClassifier,
    flattened_dataset_inputs: *const f32,
    flattened_dataset_expected_outputs: *const f32,
    len_input: usize,
    len_output: usize,
) {
    unsafe {
        let linear_model = &mut *lm;

        let input_dim = linear_model.size - 1;
        let samples_count = len_input / input_dim;

        let mut vX = DVector::<f32>::zeros(len_input);
        for i in 0..len_input {
            vX[i] = *flattened_dataset_inputs.offset(i as isize);
        }
        let X = DMatrix::from_fn(len_input, 1, |i, _| vX[i]);

        let mut vY = DVector::<f32>::zeros(len_output);
        for i in 0..len_output {
            vY[i] = *flattened_dataset_expected_outputs.offset(i as isize);
        }
        let Y = DMatrix::from_fn(len_output, 1, |i, _| vY[i]);

        let ones = DMatrix::<f32>::repeat(samples_count, 1, 1.0);

        let mut Xi = DMatrix::<f32>::zeros(X.nrows(), X.ncols() + ones.ncols());
        Xi.column_mut(0).copy_from(&X.column(0));

        let ones_col = ones.column(0);
        for i in 0..samples_count {
            Xi.column_mut(i + X.ncols()).copy_from(&ones_col);
        }

        let W = Xi.transpose() * &Xi;

        let W_inv = W.try_inverse().expect("Matrix is not invertible");

        let W2 = &W_inv * Xi.transpose();

        let W3 = &W2 * Y;

        let Ww = W3.iter().cloned().collect::<Vec<f32>>();

        for i in 0..(linear_model.size - 1) as usize {
            (*linear_model.weights.offset(i as isize)) = Ww[i];
        }
    }
}


#[no_mangle]
pub extern "C" fn predict_regression(lm: *const LinearClassifier, inputs: *const f32, inputs_size: usize) -> f32 {
    let model = unsafe { &*lm };
    let weights_slice = unsafe { std::slice::from_raw_parts(model.weights, model.size) };
    let inputs_slice = unsafe { std::slice::from_raw_parts(inputs, inputs_size - 1) };


    if model.size != inputs_size {
        panic!("Erreur de dimension");
    }

    let mut z = weights_slice[0] * 1.;


    for i in 1..model.size {
        z += weights_slice[i] * inputs_slice[i - 1];
    }
    return z;
}


#[no_mangle]
pub extern "C" fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[no_mangle]
pub extern "C" fn delete_model(lm: *mut LinearClassifier) -> Box<LinearClassifier> {
    unsafe {
        delete_float_array((*lm).weights, (*lm).size);
        Box::from_raw(lm)
    }
}


#[no_mangle]
pub extern "C" fn delete_float_array(arr: *mut f32, arr_len: usize) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len, arr_len)
    };
}
