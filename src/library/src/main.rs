use crate::modele_linaire::new;

mod modele_linaire;



fn main() {
    // Création d'un nouveau modèle avec 3 caractéristiques
    let model = new(3);
    println!(model);
    /*

    let inputs = vec![1.0, 2.0, 3.0];
    let prediction = LinearClassifier::predict(unsafe { model },  &inputs);
    unsafe { println!("Prediction: {}", *prediction); }


    let training_data = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let labels = vec![0.0, 1.0, 0.0];
    //  let accuracy = fit(training_data, labels);
    // println!("Accuracy: {}", accuracy);

    // Libération de la mémoire occupée par le modèle
    delete_model(model);
    */
}
