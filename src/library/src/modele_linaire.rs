use ndarray::{Array, Array1, Array2, s};
use rand::{thread_rng, Rng};

#[no_mangle]
pub extern "C" fn  sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Tableau pour calculer les poids avec la fonctions sigmoid
#[no_mangle]
pub extern "C" fn predict(x: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let z = x.dot(weights.slice(s![..-1])) + weights[weights.len() - 1];
    sigmoid(z)
}
