from flask import Flask, render_template, request, flash, g
from PIL import Image
from io import BytesIO
import numpy as np
from src import pmc, linear_classifier
import atexit
import secrets

app = Flask(__name__)

app.config['SECRET_KEY'] = secrets.token_urlsafe(5)

linear_classifier_class = linear_classifier.LinearClassifierModel()
pmc_class = pmc.PerceptronMC()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        image = request.files['image']
        prediction_type = request.form.get('prediction_type')
        model_type = request.form.get('model_type')
        img_bytes = BytesIO(image.read())
        img = Image.open(img_bytes)
        pixels = list(img.getdata())
        flattened_inputs = np.array(
            list(map(lambda x: [x[0], x[1], x[2]], pixels))
        ).flatten().astype(np.float32)

        if model_type == 'linear':
            if prediction_type == 'reg':
                pred = linear_classifier_class.predict_classification(g.lc_model, flattened_inputs,
                                                                      len(flattened_inputs))
            if prediction_type == 'classification':
                pred = linear_classifier_class.predict_classification(g.lc_model, flattened_inputs,
                                                                      len(flattened_inputs))
        elif model_type == 'pmc':
            if prediction_type == 'reg':
                pred = pmc_class.predict_classification(g.pmc_model, flattened_inputs, len(flattened_inputs))
            if prediction_type == 'classification':
                pred = pmc_class.predict_classification(g.pmc_model, flattened_inputs, len(flattened_inputs))

        flash(f"Prediction : {pred}")
    else:
        flash("Aucun fichier sélectionné")
    return render_template("index.html")


@app.route('/load_model/', methods=['POST'])
def load_model():
    file = request.files['model']
    type_modele = request.form.get('selected_model')
    path = file.filename
    if type_modele == 'modele_lineaire':
        g.lc_model = linear_classifier_class.load_linear_model(path)
    elif type_modele == 'pmc':
        g.pmc_model = pmc_class.load_pmc_model(path)
    flash("Modèle chargé avec succès")
    return render_template("index.html")


def deallocate_models_before_shutdown():
    if g.lc_model is not None:
        linear_classifier_class.delete_linear_model(g.lc_model)
    if g.pmc_model is not None:
        pmc_class.delete_pmc_model(g.pmc_model)


atexit.register(deallocate_models_before_shutdown)

if __name__ == "__main__":
    app.run(debug=True)
