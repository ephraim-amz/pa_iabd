use std::arch::x86_64::_mm_broadcastd_epi32;
use rand::{Rng, thread_rng};
use std::f32;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::error::Error;
use std::io::{Read, Write, BufWriter};
use libc::{c_char, puts};
use serde_derive::{Deserialize, Serialize};
use serde_json;

#[repr(C)]
#[derive(Debug, Deserialize, Serialize)]
pub struct PMC {
    layers: usize,
    dimensions: Vec<i64>,
    W: Vec<Vec<Vec<f32>>>,
    X: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
}


#[no_mangle]
pub extern "C" fn new_pmc(dimensions_arr: *const i64, layer_size_per_neuron: usize) -> *mut PMC {
    let dimensions_arr_slice =
        unsafe { std::slice::from_raw_parts(dimensions_arr, layer_size_per_neuron) };
    let mut rng = thread_rng();


    let mut pmc_model = Box::new(PMC {
        layers: (layer_size_per_neuron - 1),
        dimensions: dimensions_arr_slice.to_vec(),
        W: Vec::new(),
        X: Vec::new(),
        deltas: Vec::new(),
    });

    for layer in 0..layer_size_per_neuron {
        pmc_model.W.push(Vec::new());
        if layer != 0 {
            for weight in 0..(dimensions_arr_slice[layer - 1] + 1) as usize {
                pmc_model.W[layer].push(vec![0.0; (dimensions_arr_slice[layer] + 1) as usize]);
                for index in 0..(dimensions_arr_slice[layer] + 1) as usize {
                    pmc_model.W[layer][weight][index] = if index == 0 { 0.0 } else { rng.gen_range(-1.0..=1.) };
                }
            }
        }
    }

    for layer in 0..layer_size_per_neuron {
        let neuron_count = dimensions_arr_slice[layer] as usize + 1;
        pmc_model.X.push(vec![1.0; neuron_count]);
        pmc_model.deltas.push(vec![0.0; neuron_count]);
    }

    let leaked = Box::into_raw(pmc_model);
    leaked
}


fn forward_pass(model: *mut PMC, mut sample_inputs: Vec<f32>, is_classification: bool) {
    unsafe {
        let layers = (*model).dimensions.len();
        let input_dimensions = (*model).dimensions[0] as usize;

        for input in sample_inputs {
            let input_k = vec![input];
            for j in 0..input_dimensions {
                (*model).X[0][j] = input_k[0];
            }

            for layer in 1..layers {
                for j in 1..=(*model).dimensions[layer] as usize {
                    let mut res = 0.0;
                    for i in 0..((*model).dimensions[layer - 1] + 1) as usize {
                        res += (*model).W[layer][i][j] * (*model).X[layer - 1][i];
                    }
                    if is_classification || layer < layers - 1 {
                        res = f32::tanh(res);
                    }

                    (*model).X[layer][j] = res;
                }
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn train_pmc_model(
    model: *mut PMC,
    flattened_dataset_inputs: *const f32,
    dataset_inputs_size: usize,
    flattened_dataset_outputs: *const f32,
    dataset_outputs_size: usize,
    alpha: f32,
    epochs: i32,
    is_classification: bool,
) {
    unsafe {
        let inputs_slice = std::slice::from_raw_parts(flattened_dataset_inputs, dataset_inputs_size);
        let output_slice = std::slice::from_raw_parts(flattened_dataset_outputs, dataset_outputs_size);
        for epoch in 0..epochs {
            let mut rng = thread_rng();

            let k = rng.gen_range(0..dataset_inputs_size);
            let input_k = inputs_slice[k];
            let y_k = output_slice[k % dataset_outputs_size];

            forward_pass(model, vec![input_k], is_classification);

            for j in 1..=(*model).dimensions[(*model).layers] as usize {
                (*model).deltas[(*model).layers][j] = (*model).X[(*model).layers][j] - y_k;
                if is_classification {
                    (*model).deltas[(*model).layers][j] *= 1. - (*model).X[(*model).layers][j].powf(2.);
                }
            }

            for layer in (1..=(*model).layers).rev() {
                for i in 1..=(*model).dimensions[(*model).layers] as usize {
                    let mut res = 0.;
                    for j in 1..=(*model).dimensions[layer] as usize {
                        res += (*model).W[layer][i][j] * (*model).deltas[layer][j];
                    }
                    (*model).deltas[layer - 1][i] = (1. - (*model).X[layer - 1][i].powf(2.)) * res;
                }
            }

            for layer in 1..=(*model).layers {
                for i in 0..=(*model).dimensions[layer - 1] as usize {
                    for j in 1..=(*model).dimensions[layer] as usize {
                        (*model).W[layer][i][j] -= alpha * (*model).X[layer - 1][i] * (*model).deltas[layer][j];
                    }
                }
            }

            if epoch % 100 == 0 {
                let informations_string = format!("Epoch : {:?} Loss : {:?}", epoch, calculate_loss(flattened_dataset_inputs, dataset_inputs_size, flattened_dataset_outputs, dataset_outputs_size, is_classification));
                let informations_cstring = CString::new(informations_string).unwrap();
                puts(informations_cstring.as_ptr());
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn predict_pmc_model(
    model: *mut PMC,
    sample_inputs: *const f32,
    is_classification: bool,
) -> *mut f32 {
    unsafe {
        let input_k = sample_inputs.as_ref().unwrap();
        let mut sample_vec = vec![*input_k];
        forward_pass(model, sample_vec, is_classification);
        let informations_string = format!("Prediction output {:?}", (*model).X[(*model).layers][1..].to_vec());
        let informations_cstring = CString::new(informations_string).unwrap();
        puts(informations_cstring.as_ptr());
        return (*model).X[(*model).layers][1..].as_mut_ptr();
    }
}

#[no_mangle]
pub extern "C" fn get_X_len(model: *mut PMC) -> i32 {
    (unsafe {
        (*model).X[(*model).X.len() - 1].len() - 1
    }) as i32
}

#[no_mangle]
pub extern "C" fn delete_pmc_model(model: *mut PMC) {
    unsafe {
        drop(Box::from_raw(model));
    }
}

fn calculate_loss(
    flattened_predictions_inputs: *const f32,
    predictions_inputs_size: usize,
    flattened_labels_outputs: *const f32,
    labels_output_size: usize,
    is_classification: bool,
) -> f32 {
    unsafe {
        let predictions_slice = std::slice::from_raw_parts(flattened_predictions_inputs, predictions_inputs_size);
        let outputs_slice = std::slice::from_raw_parts(flattened_labels_outputs, labels_output_size);

        let mut loss = 0.0;

        if is_classification {
            for pred in 0..predictions_slice.len() {
                for label in 0..outputs_slice.len() {
                    let y = outputs_slice[label];
                    let p = predictions_slice[pred];
                    loss += -((y * p.ln()) + (1.0 - y) * (1.0 - p).ln());
                }
            }
        } else {
            for pred in 0..predictions_slice.len() {
                for label in 0..outputs_slice.len() {
                    loss += (predictions_slice[pred] - outputs_slice[label]).powi(2);
                }
            }
        }
        loss / predictions_slice.len() as f32
    }
}

fn get_portion_from_pointer(
    flattened_dataset_inputs: *const f32,
    input_dimensions: i64,
    k: i64,
) -> (&'static [f32], usize) {
    let start_index = (k * input_dimensions) as usize;
    let flattened_inputs_ptr: *const f32 = flattened_dataset_inputs;

    unsafe {
        let slice = std::slice::from_raw_parts(flattened_inputs_ptr.add(start_index), input_dimensions as usize);
        (slice, input_dimensions as usize)
    }
}

#[no_mangle]
pub extern "C" fn save_pmc_model(model: *mut PMC, filename: *const c_char) -> Result<(), Box<dyn Error>> {
    let m = unsafe { &*model };
    let serialized_pmc = serde_json::to_string(m)?;
    let file = unsafe { File::create(CStr::from_ptr(filename).to_str()?).expect("Une erreur est survenue lors de la création du fichier") };
    let mut buf_writer = BufWriter::new(file);
    buf_writer.write_all(serialized_pmc.as_bytes())?;
    Ok(())
}

#[no_mangle]
pub extern "C" fn load_pmc_model(path: *const c_char) -> Result<*mut PMC, Box<dyn Error>> {
    let mut file = unsafe { File::open(CStr::from_ptr(path).to_str()?)? };
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    let pmc: PMC = serde_json::from_str(&content)?;
    let leaked = Box::into_raw(Box::new(pmc));
    Ok(leaked)
}



