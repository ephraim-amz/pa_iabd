use std::io::stderr;
use ndarray::{Array1, s};

// Tableau pour calculer les poids avec la fonctions sigmoid

pub struct LinearModel {
    X: Vec<f32>,
}

#[no_mangle]
pub extern "C" fn create_linear_model(x: Vec<f32>) -> LinearModel {
    LinearModel{
        X: x
    }
}

#[no_mangle]
pub extern "C" fn delete_array(vector: *mut vec<f32>, size: usize) {
    unsafe {
        Vec::from_raw_parts(vector, size as usize, size as usize)
    };
}




#[no_mangle]
pub extern "C" fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}


#[no_mangle]
pub extern "C" fn destroy_model(lm: LinearModel){
    delete_array(lm.X)
}



#[no_mangle]
pub extern "C" fn predict(lm: LinearModel, weights: &Vec<f32>) -> f32 {
    let mut z: f32 = 0.0;
    if lm.X.len() != weights.len() {
        panic!("Erreur de dimension");
    }
    for i in 0..lm.X.len() - 1 {
        z += lm.X.get(i).unwrap() * weights.get(i).unwrap()
    }
    z += weights.get(weights.len() - 1).unwrap();
    sigmoid(z)
}




#[no_mangle]
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}


/*

#[no_mangle]
pub extern "C" fn predict_two(x: &[f64], weights: &[f64]) -> f64 {
    let mut z = 0.;
    for i in 0..x.len() {
        z += x[i] * weights[..weights.len() - 1] + weights[weights.len() - 1]
    }
    sigmoid(z)
}


 */

