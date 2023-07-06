use std::slice;
use rand::Rng;

#[repr(C)]
pub struct PMC {
    layers: usize,
    neurons_per_layer: Vec<i32>,
    W: Vec<Vec<Vec<f32>>>,
    X: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
}

#[no_mangle]
pub extern "C" fn new_pmc(neurons_per_layer: *const i32, layer_size_per_neuron: usize) -> *mut PMC {
    let neurons_per_layer_slice = unsafe { slice::from_raw_parts(neurons_per_layer, layer_size_per_neuron) };
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
pub extern "C" fn predict_pmc_model(model: *mut PMC, sample_inputs: *const f32, columns: i32,
                                    is_classification: bool) -> *mut f32 {
    unimplemented!()
}

#[no_mangle]
pub extern "C" fn delete_pmc_model(model: *mut PMC) {
    unsafe {
        drop(Box::from_raw(model));
    }
}