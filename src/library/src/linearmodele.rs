
use std::vec;
use rand::Rng;


#[no_mangle]
pub extern "C" fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}



struct LinearClassifier {
    weights: vec<f32>,
    bias: f32,
}
// Definition de la structure pour les poids

impl LinearClassifier {
    fn new(num_features: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f32> = (0..num_features).map(|_| rng.gen_range(-1.0, 1.0)).collect();
        let bias = 0.0;
        LinearClassifier { weights, bias }
    }
}

#[no_mangle]
pub extern "C" fn predict(lm: LinearClassifier, weights: &Vec<f32>) -> f32 {
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



// faire la fonction fit et prédict