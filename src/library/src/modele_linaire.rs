use rand::Rng;


#[repr(C)]
pub struct LinearClassifier {
    pub weights: *mut f32,
    pub size: usize,
}


#[no_mangle]
pub extern "C" fn new(num_features: usize) -> *mut LinearClassifier {
    let mut rng = rand::thread_rng();
    let model = {
        let weights = Box::into_raw(vec![rng.gen_range(-1.0..1.0); num_features].into_boxed_slice()) as *mut f32;
        let lm = LinearClassifier {
            size: num_features,
            weights,
        };
        Box::into_raw(Box::new(lm))
    };
    let leak_lm = Box::leak(lm);
    leak_lm
}

/*


#[no_mangle]
pub extern "C" fn fit(
    model: *mut LinearClassifier,
    flattened_inputs: *mut f32,
    flattened_outputs: *const f32,
    input_size: i32,
    output_size: i32,
) {
    unsafe {
        let dimensions = (*model).weights.len() - 1;
        let samples_count = output_size / dimensions as i32;


        let mut x_matrix: Vec<Vec<f32>> = vec![vec![1.; dimensions]; dimensions];
        let mut x_vector = vec![0.; dimensions];

        // let flattened_input_slice = std::slice::from_raw_parts(flattened_input, dimension);
        // let flattened_output_slice = std::slice::from_raw_parts(flattened_output, dimension);


        for i in 0..input_size {
            &x_vector[i as usize] = &flattened_inputs[i as usize]
        }

        let y_vector = vec![0., dimensions];

        for i in 0..input_size {
            y_vector.get(i).unwrap() = flattened_outputs.get(i).unwrap();
        }

        // TODO : Calcul matriciel avec la méthode du pseudo inverse
    }
}

*/


#[no_mangle]
pub extern "C" fn predict(lm: *const LinearClassifier, inputs: *const f32, inputs_size: usize) -> f32 {
    let model = unsafe { &*lm };
    let weights_slice = unsafe { std::slice::from_raw_parts(model.weights, model.size - 1) };
    let inputs_slice = unsafe { std::slice::from_raw_parts(inputs, inputs_size - 1) };

    if model.size != inputs_size {
        panic!("Erreur de dimension");
    }

    let mut z = weights_slice[0] * 1.;


    for i in 0..=model.size - 1 {
        z += weights_slice[i + 1] * inputs_slice[i];
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
        Vec::from_raw_parts(arr, arr_len as usize, arr_len)
    };
}
