use std::error::Error;
use std::ffi::CStr;
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use libc::{c_char, puts};
use rand::Rng;
use nalgebra::DMatrix;
use rand::distributions::Uniform;
use serde_derive::{Serialize, Deserialize};
use serde_json;

#[repr(C)]
#[derive(Debug, Deserialize, Serialize)]
pub struct LinearClassifier {
    weights: Vec<f32>,
    size: usize,
}

#[no_mangle]
pub extern "C" fn new(num_features: usize) -> *mut LinearClassifier {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new_inclusive(-1.0, 1.0);
    let weights: Vec<f32> = (0..num_features + 1).map(|_| rng.sample(dist)).collect();
    let model = Box::new(LinearClassifier {
        size: num_features,
        weights,
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

        let input_dim = linear_model.size;
        let samples_count = len_input / input_dim;

        let X = DMatrix::from_row_slice(samples_count, input_dim, std::slice::from_raw_parts(flattened_dataset_inputs, len_input));
        let Y = DMatrix::from_row_slice(len_output, 1, std::slice::from_raw_parts(flattened_dataset_expected_outputs, len_output));
        let ones = DMatrix::<f32>::repeat(samples_count, 1, 1.);

        /*let x_matrix_string = format!("X {:?}", X);
        let x_matrix_cstring = CString::new(x_matrix_string).unwrap();
        puts(x_matrix_cstring.as_ptr());

        let y_matrix_string = format!("Y {:?}", Y);
        let y_matrix_cstring = CString::new(y_matrix_string).unwrap();
        puts(y_matrix_cstring.as_ptr());*/

        let mut Xi = DMatrix::<f32>::zeros(X.nrows(), X.ncols() + ones.ncols());

        Xi.column_mut(0).copy_from(&X.column(0));

        let first_one_column = ones.column(0);

        for i in 0..samples_count {
            Xi.column_mut(i + X.ncols()).copy_from(&first_one_column);
        }

        let W = Xi.transpose() * &Xi;
        let inv_W = W.try_inverse().expect("Matrice non inversible");
        let W2 = &inv_W * Xi.transpose();
        let W3 = &W2 * Y;

        linear_model.weights = W3.iter().cloned().collect::<Vec<f32>>();
    }
}

#[no_mangle]
pub extern "C" fn train_classification(
    lm: *mut LinearClassifier,
    flattened_inputs: *const f32,
    flattened_outputs: *const f32,
    inputs_size: usize,
    output_size: usize,
    lr: f32,
    epochs: i32,
) {
    let mut rng = rand::thread_rng();

    unsafe {
        let inputs = std::slice::from_raw_parts(flattened_inputs, inputs_size);
        let outputs = std::slice::from_raw_parts(flattened_outputs, output_size);

        for _ in 0..epochs {
            let k = rng.gen_range(0..inputs_size / output_size); // Voir la taille du k qui peut être erroné
            let inputs_slice = &inputs[k * output_size..(k + 1) * output_size];
            let mut Xk = vec![1.0];
            Xk.extend_from_slice(inputs_slice);


            let gXk = dot_product(&(*lm).weights, &Xk); // Voir usage de sigmoid ou tanh


            for (i, weight) in (*lm).weights.iter_mut().skip(1).enumerate() {
                *weight += lr * (outputs[k] - gXk) * Xk[i];
            }
        }
    }
}

fn dot_product(weights: &Vec<f32>, inputs: &Vec<f32>) -> f32 {
    weights.iter().zip(inputs).map(|(w, x)| w * x).sum()
}


#[no_mangle]
pub extern "C" fn predict_regression(
    lm: *const LinearClassifier,
    inputs: *const f32,
    inputs_size: usize,
) -> f32 {
    let model = unsafe { &*lm };
    let inputs_slice = unsafe { std::slice::from_raw_parts(inputs, inputs_size) };

    let mut z = model.weights[0];

    for i in 1..=model.size {
        z += model.weights[i] * inputs_slice[i - 1];
    }
    z
}

#[no_mangle]
pub extern "C" fn predict_classification(
    lm: *const LinearClassifier,
    inputs: *const f32,
    inputs_size: usize,
) -> f32 {
    let pred_value = predict_regression(lm, inputs, inputs_size);
    if pred_value > 0.0 {
        1.0
    } else if pred_value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[no_mangle]
pub extern "C" fn delete_linear_model(lm: *mut LinearClassifier) {
    unsafe {
        if !lm.is_null() {
            drop(Box::from_raw(lm));
        }
    }
}

#[no_mangle]
pub extern "C" fn save_linear_model(model: *mut LinearClassifier, filename: *const c_char) -> Result<(), Box<dyn Error>> {
    let m = unsafe { &*model };
    let serialized_pmc = serde_json::to_string(m)?;
    let file = unsafe { File::create(CStr::from_ptr(filename).to_str()?).expect("Une erreur est survenue lors de la création du fichier") };
    let mut buf_writer = BufWriter::new(file);
    buf_writer.write_all(serialized_pmc.as_bytes())?;
    Ok(())
}

#[no_mangle]
pub extern "C" fn load_linear_model(path: *const c_char) -> Result<*mut LinearClassifier, Box<dyn Error>> {
    let mut file = unsafe { File::open(CStr::from_ptr(path).to_str()?)? };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let lm: LinearClassifier = serde_json::from_str(&content)?;
    let leaked = Box::into_raw(Box::new(lm));
    Ok(leaked)
}
