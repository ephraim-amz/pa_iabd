/*
use std::vec::Vec;
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;

#[repr(C)]
pub struct MLP {
    W: Vec<Vec<Vec<f32>>>,
    d: Vec<i32>,
    X: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
}

#[no_mangle]
pub extern "C" fn create_mlp_model(npl: *const i32, npl_size: usize) -> Box<MLP> {
    let mut rng = rand::thread_rng();
    let mut model = Box::new(MLP {
        W: Vec::new(),
        d: Vec::new(),
        X: Vec::new(),
        deltas: Vec::new(),
    });

    let npl_slice = unsafe { std::slice::from_raw_parts(npl, npl_size) };

    for l in 0..npl_size {
        model.W.push(Vec::new());
        if l != 0 {
            for _i in 0..(npl_slice[l - 1] + 1) {
                model.W[l].push(vec![0.0; (npl_slice[l] + 1) as usize]);
                for j in 0..(npl_slice[l] + 1) {
                    model.W[l][_i as usize][j as usize] = rng.gen_range(-1.0..=1.0);
                }
            }
        }
    }

    for i in 0..npl_size {
        model.d.push(npl_slice[i]);
    }

    for l in 0..npl_size {
        model.X.push(Vec::new());
        for j in 0..(npl_slice[l] + 1) {
            if j == 0 {
                model.X[l].push(1.0);
            } else {
                model.X[l].push(0.0);
            }
        }
    }

    for l in 0..npl_size {
        model.deltas.push(Vec::new());
        for _j in 0..(npl_slice[l] + 1) {
            model.deltas[l].push(0.0);
        }
    }

    model
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_regression(model: &mut MLP, sample_inputs: *const f32) -> *mut f32 {
    let last_index = model.X.len() - 1;
    let size = model.X[last_index].len() - 1;
    let converted_vector = Box::into_raw(vec![0.0; size as usize].into_boxed_slice()) as *mut f32;

    unsafe {
        model.forward_pass(std::slice::from_raw_parts(sample_inputs, model.d[0] as usize), false);
        for i in 0..size {
            *converted_vector.add(i as usize) = model.X[last_index][(i + 1) as usize];
        }
    }

    converted_vector
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_classification(model: &mut MLP, sample_inputs: *const f32) -> *mut f32 {
    let last_index = model.X.len() - 1;
    let size = model.X[last_index].len() - 1;
    let converted_vector = Box::into_raw(vec![0.0; size as usize].into_boxed_slice()) as *mut f32;

    unsafe {
        model.forward_pass(std::slice::from_raw_parts(sample_inputs, model.d[0] as usize), true);
        for i in 0..size {
            *converted_vector.add(i as usize) = model.X[last_index][(i + 1) as usize];
        }
    }

    converted_vector
}

#[no_mangle]
pub extern "C" fn train_classification_stochastic_gradient_backpropagation_mlp_model(
    model: &mut MLP,
    flattened_dataset_inputs: *mut f32,
    flattened_dataset_inputs_size: usize,
    flattened_dataset_expected_outputs: *mut f32,
    alpha: f32,
    iterations_count: i32,
) {
    let sample_count = flattened_dataset_inputs_size / model.d[0] as usize;
    let last_index = model.d.len() - 1;

    for _it in 0..iterations_count {
        let k = rand::thread_rng().gen_range(0..sample_count);
        let sample_input = unsafe {
            std::slice::from_raw_parts(
                flattened_dataset_inputs.add((k * model.d[0] as usize) as usize),
                model.d[0] as usize,
            )
        };
        let sample_expected_output = unsafe {
            std::slice::from_raw_parts(
                flattened_dataset_expected_outputs.add((k * model.d[last_index] as usize) as usize),
                model.d[last_index] as usize,
            )
        };

        model.forward_pass(sample_input, true);

        for j in 1..=model.d[last_index] {
            model.deltas[last_index][j as usize] =
                model.X[last_index][j as usize] - sample_expected_output[(j - 1) as usize];
            if true /* is_classification */ {
                model.deltas[last_index][j as usize] *= 1.0 - model.X[last_index][j as usize].powi(2);
            }
        }

        for l in (1..last_index).rev() {
            for i in 1..=model.d[l - 1] {
                let mut sum_result = 0.0;
                for j in 1..=model.d[l] {
                    sum_result += model.W[l][i as usize][j as usize] * model.deltas[l + 1][j as usize];
                }
                model.deltas[l][i as usize] = (1.0 - model.X[l][i as usize].powi(2)) * sum_result;
            }
        }

        for l in 1..=last_index {
            for i in 0..=model.d[l - 1] {
                for j in 1..=model.d[l] {
                    model.W[l][i as usize][j as usize] -=
                        alpha * model.X[l - 1][i as usize] * model.deltas[l][j as usize];
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn train_regression_stochastic_gradient_backpropagation_mlp_model(
    model: &mut MLP,
    flattened_dataset_inputs: *mut f32,
    flattened_dataset_inputs_size: usize,
    flattened_dataset_expected_outputs: *mut f32,
    alpha: f32,
    iterations_count: i32,
) {
    let sample_count = flattened_dataset_inputs_size / model.d[0] as usize;
    let last_index = model.d.len() - 1;

    for _it in 0..iterations_count {
        let k = rand::thread_rng().gen_range(0..sample_count);
        let sample_input = unsafe {
            std::slice::from_raw_parts(
                flattened_dataset_inputs.add((k * model.d[0] as usize) as usize),
                model.d[0] as usize,
            )
        };
        let sample_expected_output = unsafe {
            std::slice::from_raw_parts(
                flattened_dataset_expected_outputs.add((k * model.d[last_index] as usize) as usize),
                model.d[last_index] as usize,
            )
        };

        model.forward_pass(sample_input, false);

        for j in 1..=model.d[last_index] {
            model.deltas[last_index][j as usize] =
                model.X[last_index][j as usize] - sample_expected_output[(j - 1) as usize];
            // No need to calculate deltas for regression
        }

        for l in (1..last_index).rev() {
            for i in 1..=model.d[l - 1] {
                let mut sum_result = 0.0;
                for j in 1..=model.d[l] {
                    sum_result += model.W[l][i as usize][j as usize] * model.deltas[l + 1][j as usize];
                }
                model.deltas[l][i as usize] = (1.0 - model.X[l][i as usize].powi(2)) * sum_result;
            }
        }

        for l in 1..=last_index {
            for i in 0..=model.d[l - 1] {
                for j in 1..=model.d[l] {
                    model.W[l][i as usize][j as usize] -=
                        alpha * model.X[l - 1][i as usize] * model.deltas[l][j as usize];
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn get_x_size(model: &MLP) -> i32 {
    (model.X.last().unwrap().len() - 1) as i32
}

*/
