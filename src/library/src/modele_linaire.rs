use ndarray::{Array, Array1, Array2, s};
use rand::{thread_rng, Rng};


// Tableau pour calculer les poids avec la fonctions sigmoid


#[no_mangle]
pub extern "C" fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
/*
#[no_mangle]
pub extern "C" fn predict(x: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let z = x.dot(weights.slice(s![..-1]).as_slice().unwrap()) + weights[weights.len() - 1];
    sigmoid(z)
}
*/

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

