from flask import Flask, render_template, request, flash, g
from PIL import Image
from io import BytesIO
import numpy as np
import linear_classifier
import pmc
import atexit
import secrets

app = Flask(__name__)

app.config['SECRET_KEY'] = secrets.token_urlsafe(5)

linear_classifier_class = linear_classifier.LinearClassifierModel()
pmc_class = pmc.PerceptronMC()


# lc_model = linear_classifier_class.load_linear_model("linear_model.json")


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        img_bytes = BytesIO(image.read())
        img = Image.open(img_bytes)
        pixels = list(img.getdata())
        flattened_inputs = np.array(
            list(map(lambda x: [x[0], x[1], x[2]], pixels))
        ).flatten().astype(np.float32)

        flash("Image téléchargée avec succès et convertie en pixels")
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
        g.pmc_model = pmc_class.new()
    flash("Modèle chargé avec succès")
    return render_template("index.html")


"""
def train_image(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    flattened_inputs = np.array(
        list(map(lambda x: [x[0], x[1], x[2]], list(img.getdata())[0:10]))
    ).flatten().astype(np.float32)

    lib.train_classification(flattened_inputs, [1, 1, -1])
"""


def deallocate_models_before_shutdown():
    linear_classifier_class.delete_pmc_model(g.lc_model)


atexit.register(deallocate_models_before_shutdown)

if __name__ == "__main__":
    app.run(debug=True)
