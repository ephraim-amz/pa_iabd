use ndarray::s;
use rand::Rng;
use std::f32;

#[repr(C)]
pub struct PMC {
    layers: usize,
    neurons_per_layer: Vec<i32>,
    W: Vec<Vec<Vec<f32>>>,
    X: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
}

fn forward_pass(model: *mut PMC, sample_inputs: *const f32, sample_inputs_size: usize, is_classification: bool) {
    unsafe {
        let sample_inputs_slice = std::slice::from_raw_parts(sample_inputs, sample_inputs_size);
        let layers = (*model).neurons_per_layer.len() - 1;
        for j in 1..=(*model).neurons_per_layer[0] as usize {
            (*model).X[0][j] = sample_inputs_slice[j - 1];
        }
        for layer in 1..=layers {
            for j in 1..(*model).neurons_per_layer[layer] as usize {
                let mut res = 0.;
                for i in 0..=(*model).neurons_per_layer[layer - 1] as usize {
                    res += (*model).W[layer][i][j] * (*model).X[layer - 1][i];
                }
                (*model).X[layer][j] = res;
                if is_classification || layer < layers {
                    (*model).X[layer][j] = f32::tanh((*model).X[layer][j]);
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn new_pmc(neurons_per_layer: *const i32, layer_size_per_neuron: usize) -> *mut PMC {
    let neurons_per_layer_slice = unsafe { std::slice::from_raw_parts(neurons_per_layer, layer_size_per_neuron) };
    let mut rng = rand::thread_rng();

    let mut pmc_model = Box::new(PMC {
        layers: 0,
        neurons_per_layer: neurons_per_layer_slice.to_vec(),
        W: Vec::new(),
        X: Vec::new(),
        deltas: vec![vec![0.0; layer_size_per_neuron]],
    });


    for layer in 0..layer_size_per_neuron {
        pmc_model.W.push(Vec::new());
        if layer != 0 {
            for weight in 0..(neurons_per_layer_slice[layer - 1] + 1) as usize {
                pmc_model.W[layer].push(vec![0.0; (neurons_per_layer_slice[layer] + 1) as usize]);
                for index in 0..(neurons_per_layer_slice[layer] + 1) as usize {
                    pmc_model.W[layer][weight][index] = rng.gen_range(-1.0..=1.)
                }
            }
        }
    }

    for layer in 0..layer_size_per_neuron {
        pmc_model.X.push(Vec::new());
        for weight in 0..(neurons_per_layer_slice[layer] + 1) {
            if weight != 0 {
                pmc_model.X[layer].push(0.);
            } else {
                pmc_model.X[layer].push(1.);
            }
        }
    }

    let leaked = Box::leak(pmc_model);
    leaked
}

#[no_mangle]
pub extern "C" fn train_pmc_model(model: *mut PMC, dataset_inputs: *const f32, lines: i32, columns: i32,
                                  dataset_outputs: *const f32, output_columns: i32, alpha: f32, nb_iter: i32,
                                  is_classification: bool) {
    unimplemented!()
}

#[no_mangle]
pub extern "C" fn predict_pmc_model(
    model: *mut PMC,
    sample_inputs: *const f32,
    sample__inputs_size: usize,
    is_classification: bool,
) -> *mut f32 {
    return if is_classification {
        predict_pmc_classification(model, sample_inputs, sample__inputs_size)
    } else {
        predict_pmc_regression(model, sample_inputs, sample__inputs_size)
    };
}

#[no_mangle]
pub extern "C" fn predict_pmc_classification(
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
            *prediction_vector.add(i) = (*model).X[last_element][(i + 1)];
        }
        prediction_vector
    }
}

#[no_mangle]
pub extern "C" fn predict_pmc_regression(
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
            *prediction_vector.add(i) = (*model).X[last_element][(i + 1)];
        }
        prediction_vector
    }
}

#[no_mangle]
pub extern "C" fn delete_pmc_model(model: *mut PMC) {
    unsafe {
        drop(Box::from_raw(model));
    }
}