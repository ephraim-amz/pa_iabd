use rand::Rng;

// Definition de la structure pour les poids

#[repr(C)]
pub struct LinearClassifier {
    pub weights: Vec<f32>,
    pub bias: f32,
}

#[no_mangle]
pub extern "C" fn new(num_features: usize) -> *mut LinearClassifier {
    let mut rng = rand::thread_rng();
    let weights: Vec<f32> = (0..num_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let bias = 0.0;
    let model: Box<LinearClassifier> = Box::new(LinearClassifier { weights, bias });
    let leaked_model = Box::leak(model);
    leaked_model
}

impl LinearClassifier {
    #[no_mangle]
    pub extern "C" fn fit(X: Vec<Vec<f32>>, y: Vec<f32>) -> f32 {
        unimplemented!()
    }

    #[no_mangle]
    pub extern "C" fn predict(lm: *mut LinearClassifier, inputs: &Vec<f32>) -> *mut f32 {
        let mut z: Vec<f32> = vec![0.];

        unsafe {
            if (*lm).weights.len() != inputs.len() {
                panic!("Erreur de dimension");
            }


            for i in 0..=(*lm).weights.len() - 1 {
                z[0] += (*lm).weights[i] * inputs[i]
            }
            z[0] += (*lm).bias;

            let mut result: Vec<f32> = vec![sigmoid(z[0])];
            result.leak().as_mut_ptr()
        }
    }
}


#[no_mangle]
pub extern "C" fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[no_mangle]
pub extern "C" fn delete_model(lm: *mut LinearClassifier) -> Box<LinearClassifier> {
    unsafe {
        Box::from_raw(lm)
    }
}


#[no_mangle]
pub extern "C" fn delete_float_array(arr: *mut f32, arr_len: i32) {
    unsafe {
        Vec::from_raw_parts(arr, arr_len as usize, arr_len as usize)
    };
}

// faire la fonction fit et prédict
