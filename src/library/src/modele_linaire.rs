use rand::Rng;

// Definition de la structure pour les poids

pub struct LinearClassifier {
    weights: Vec<f32>,
    bias: f32,
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

    /*
    #[no_mangle]
    pub extern "C" fn predict(lm: LinearClassifier, inputs: *mut Vec<f32>) -> f32 {
        let mut z: f32 = 0.0;
        if lm.X.len() != inputs.len() {
            panic!("Erreur de dimension");
        }
        for i in 0..lm.X.len() - 1 {
            z += lm.X.get(i).unwrap() * inputs.get(i).unwrap()
        }
        z += inputs.get(inputs.len() - 1).unwrap();
        sigmoid(z)
    }
    */
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