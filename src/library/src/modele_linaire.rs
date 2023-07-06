use rand::Rng;
use nalgebra::{DMatrix, DVector};

#[repr(C)]
pub struct Model {
    pub weights: *mut f32,
    pub size: usize,
}


#[no_mangle]
pub extern "C" fn new(num_features: usize) -> *mut Model {
    let mut rng = rand::thread_rng();
    let model = Box::new(Model {
        size: num_features,
        weights: Box::into_raw(vec![rng.gen_range(-1.0..1.0); num_features].into_boxed_slice()) as *mut f32,
    });
    let leak_lm = Box::leak(model);
    leak_lm
}


#[no_mangle]
pub extern "C" fn train_regression(
    lm: *mut Model,
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
pub extern "C" fn predict_regression(lm: *const Model, inputs: *const f32, inputs_size: usize) -> f32 {
    let model = unsafe { &*lm };
    let weights_slice = unsafe { std::slice::from_raw_parts(model.weights, model.size) };
    let inputs_slice = unsafe { std::slice::from_raw_parts(inputs, inputs_size - 1) };


    let mut z = weights_slice[0];

    for i in 1..model.size {
        z += weights_slice[i] * inputs_slice[i - 1];
    }
    return z;
}

#[no_mangle]
pub fn train_classification(
    lm: *mut Model,
    flattened_inputs: *const f32,
    flattened_outputs: *const f32,
    inputs_size: usize,
    output_size: usize,
    lr: f32,
    epochs: i32,
) {
    let mut rng = rand::thread_rng();

    unsafe {
        for _ in 0..epochs {
            let k = rng.gen_range(0..(*lm).size);
            let inputs = std::slice::from_raw_parts(flattened_inputs, inputs_size * output_size);
            let outputs = std::slice::from_raw_parts(flattened_outputs, output_size);

            let mut Xk = vec![1.0];
            for i in 0..output_size {
                Xk.push(inputs[k * output_size + i]);
            }

            let gXk = match dot_product(transpose((*lm).weights), &Xk) {
                x if x >= 1.0 => 1.0,
                x if x <= -1.0 => 2.0,
                _ => 0.0,
            };

            for i in 1..(*lm).size {
                let weight_ptr = (*lm).weights.add(i - 1);
                *weight_ptr = *weight_ptr + lr * (outputs[k] - gXk) * Xk[i - 1];
            }
        }
    }
}

fn dot_product(weights: *mut f32, inputs: &[f32]) -> f32 {
    let mut result = 0.0;
    for i in 0..inputs.len() {
        unsafe {
            result += *weights.add(i) * inputs[i];
        }
    }
    result
}

fn transpose(weights: *mut f32) -> *mut f32 {
    let mut transposed = Vec::new();
    for i in 0..3 {
        unsafe {
            transposed.push(*weights.add(i));
        }
    }
    transposed.as_mut_ptr()
}


#[no_mangle]
pub extern "C" fn predict_classification(lm: *const Model, inputs: *const f32, inputs_size: usize) -> f32
{
    let pred_value = predict_regression(lm, inputs, inputs_size);
    return if pred_value > 0. {
        1.
    } else if pred_value < 0.0 {
        -1.0
    } else {
        0.0
    };
}


#[no_mangle]
pub extern "C" fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[no_mangle]
pub extern "C" fn delete_model(lm: *mut Model) -> Box<Model> {
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
