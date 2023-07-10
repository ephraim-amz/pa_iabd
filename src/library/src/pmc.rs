use rand::{Rng, thread_rng};
use std::f32;

use std::os::raw::c_char;
use std::ffi::CString;
use libc::c_void;
use libc::c_int;


#[repr(C)]
pub struct PMC {
    layers: u32,
    dimensions: Vec<i64>,
    W: Vec<Vec<Vec<f32>>>,
    X: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
}


#[no_mangle]
pub extern "C" fn new_pmc(dimensions_arr: *const i64, layer_size_per_neuron: usize) -> *mut PMC {
    let dimensions_arr_slice = unsafe { std::slice::from_raw_parts(dimensions_arr, layer_size_per_neuron) };
    let mut rng = thread_rng();

    let mut pmc_model = Box::new(PMC {
        layers: (layer_size_per_neuron - 1) as u32,
        dimensions: dimensions_arr_slice.to_vec(),
        W: Vec::new(),
        X: Vec::new(),
        deltas: vec![vec![0.0; dimensions_arr_slice[layer_size_per_neuron - 1] as usize]; layer_size_per_neuron],
    });

    for layer in 0..layer_size_per_neuron {
        pmc_model.W.push(Vec::new());
        if layer != 0 {
            for weight in 0..(dimensions_arr_slice[layer - 1] + 1) as usize {
                pmc_model.W[layer].push(vec![0.0; (dimensions_arr_slice[layer] + 1) as usize]);
                for index in 0..(dimensions_arr_slice[layer] + 1) as usize {
                    pmc_model.W[layer][weight][index] = rng.gen_range(-1.0..=1.);
                }
            }
        }
    }

    for layer in 0..layer_size_per_neuron {
        let neuron_count = dimensions_arr_slice[layer] as usize + 1;
        pmc_model.X.push(vec![0.0; neuron_count]);
    }

    let leaked = Box::into_raw(pmc_model);
    leaked
}

#[no_mangle]
pub extern "C" fn train_pmc_model(
    model: *mut PMC,
    flattened_dataset_inputs: *const f32,
    dataset_inputs_size: usize,
    flattened_dataset_outputs: *const f32,
    alpha: f32,
    epochs: i32,
    is_classification: bool,
) {
    let mut rng = thread_rng();
    unsafe {
        let input_dimensions = (*model).dimensions[0];
        let last_index = (*model).dimensions.len() - 1;
        let nb_outputs = (*model).dimensions[last_index];
        let sample_count = (dataset_inputs_size as f32 / input_dimensions as f32).floor() as i32;
        let layers = (*model).layers as usize;
        for _epoch in 0..epochs {
            let k = rng.gen_range(0..=sample_count) as i64;

            let (inputs_slice, inputs_slice_length) =
                get_portion_from_pointer(flattened_dataset_inputs, input_dimensions, k);
            let (outputs_slice, _) = get_portion_from_pointer(flattened_dataset_outputs, nb_outputs, k);

            forward_pass(model, inputs_slice.as_ptr(), inputs_slice_length, is_classification);

            let predicted_outputs = (*model).X[last_index][1..].to_vec();
            let targets = outputs_slice;

            let predicted_outputs_size = predicted_outputs.len();
            let mut losses_vector = Vec::with_capacity(predicted_outputs_size);

            for i in 0..predicted_outputs_size {
                let predicted_output = predicted_outputs[i];
                let target = targets[i];
                if is_classification {
                    losses_vector.push(-target * predicted_output.log2()
                        - (-1. - target) * (-1. - predicted_output).log2());
                } else {
                    losses_vector.push((predicted_output - target).powi(2))
                }
            }

            let mut output_errors = Vec::with_capacity(predicted_outputs_size);

            for i in 0..predicted_outputs_size {
                let predicted_output = predicted_outputs[i];
                let target = targets[i];
                if is_classification {
                    output_errors.push(target - predicted_output);
                } else {
                    output_errors.push(predicted_output - target);
                }
            }

            for l in (1..=layers - 1).rev() {
                for i in 1..=(*model).dimensions[l - 1] as usize {
                    let mut res: f32 = 0.;
                    for j in 1..=(*model).dimensions[l] as usize {
                        res += (*model).W[l][i][j] * (*model).deltas[l][j];
                    }
                    (*model).deltas[l - 1][i] =
                        (1. - (*model).X[l - 1][i] * (*model).X[l - 1][i]) * res;
                }
            }

            for l in 1..layers {
                for i in 0..=(*model).dimensions[l - 1] as usize {
                    for j in 1..=(*model).dimensions[l] as usize {
                        (*model).W[l][i][j] -= alpha * (*model).X[l - 1][i] * (*model).deltas[l][j];
                    }
                }
            }
        }

        for k in 0..sample_count as i64 {
            let (inputs_slice, inputs_slice_length) =
                get_portion_from_pointer(flattened_dataset_inputs, input_dimensions, k);
            forward_pass(model, inputs_slice.as_ptr(), inputs_slice_length, is_classification);
        }
    }
}


#[no_mangle]
pub extern "C" fn predict_pmc_model(
    model: *mut PMC,
    sample_inputs: *const f32,
    sample_inputs_size: usize,
    is_classification: bool,
) -> *mut f32 {
    return if is_classification {
        predict_pmc_classification(model, sample_inputs, sample_inputs_size)
    } else {
        predict_pmc_regression(model, sample_inputs, sample_inputs_size)
    };
}

fn predict_pmc_classification(
    model: *mut PMC,
    sample_inputs: *const f32,
    sample_inputs_size: usize,
) -> *mut f32 {
    unsafe {
        let last_element = (*model).X.len() - 1;
        let size = (*model).X[last_element].len() - 1;
        let prediction_vector = Box::into_raw(vec![0.0; size].into_boxed_slice()) as *mut f32;
        forward_pass(model, sample_inputs, sample_inputs_size, true);

        for i in 0..size {
            *prediction_vector.add(i) = (*model).X[last_element][(i)];
        }
        prediction_vector
    }
}


#[no_mangle]
pub extern "C" fn get_X_len(model: *mut PMC) -> i32 {
    (unsafe {
        (*model).X[(*model).X.len() - 1].len() - 1
    }) as i32
}

fn predict_pmc_regression(
    model: *mut PMC,
    sample_inputs: *const f32,
    sample_inputs_size: usize,
) -> *mut f32 {
    unsafe {
        let last_element = (*model).X.len() - 1;
        let size = (*model).X[last_element].len() - 1;
        let prediction_vector = Box::into_raw(vec![0.0; size].into_boxed_slice()) as *mut f32;

        forward_pass(model, sample_inputs, sample_inputs_size, false);

        for i in 0..size {
            *prediction_vector.add(i) = (*model).X[last_element][(i)];
        }
        prediction_vector
    }
}


#[no_mangle]
pub extern "C" fn delete_pmc_model(model: *mut PMC) {
    unsafe {
        Box::from_raw(model);
    }
}


fn forward_pass(model: *mut PMC, sample_inputs: *const f32, sample_inputs_size: usize, is_classification: bool) {
    unsafe {
        let sample_inputs_slice = std::slice::from_raw_parts(sample_inputs, sample_inputs_size);
        let layers = (*model).dimensions.len();
        let input_dimensions = (*model).dimensions[0] as usize;
        let sample_count = sample_inputs_size / input_dimensions;

        for k in 0..sample_count {
            for j in 0..input_dimensions {
                (*model).X[0][j + 1] = sample_inputs_slice[k * input_dimensions + j];
            }

            for layer in 1..layers {
                for j in 1..=(*model).dimensions[layer] as usize {
                    let mut res = 0.0;
                    for i in 0..=(*model).dimensions[layer - 1] as usize {
                        res += (*model).W[layer][i][j] * (*model).X[layer - 1][i];
                    }
                    (*model).X[layer][j] = res;
                    if is_classification || layer < layers - 1 {
                        (*model).X[layer][j] = f32::tanh((*model).X[layer][j]);
                    }
                }
            }
        }
    }
}


fn get_portion_from_pointer(
    flattened_dataset_inputs: *const f32,
    input_dimensions: i64,
    k: i64,
) -> (&'static [f32], usize) {
    let start_index = (k * input_dimensions) as usize;
    let raw_ptr: *const f32 = flattened_dataset_inputs;

    unsafe {
        let slice = std::slice::from_raw_parts(raw_ptr.add(start_index), input_dimensions as usize);
        (slice, input_dimensions as usize)
    }
}
