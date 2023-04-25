use ndarray::{Array, Array1, Array2, s};
use rand::{thread_rng, Rng};

#[no_mangle]
pub extern "C" fn  sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
