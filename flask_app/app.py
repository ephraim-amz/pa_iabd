from flask import Flask, render_template, request
from PIL import Image
import numpy as np

app = Flask(__name__)


@app.route('/index/')
def index():
    m = 42
    return render_template("index.html", m=m)


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        image = request.files['image']
        image.save('uploads/' + image.filename)

        # ouvrir l'image avec la bib pillow
        img = Image.open('uploads/' + image.filename)

        # conversion en pixel
        pixels = list(img.getdata())
        width, height = img.size

        # ajouter les à la liste
        image_pixels = []
        for pixel in pixels:
            # Ajouter le pixel à la liste des pixels de l'image
            image_pixels.append(pixel)

        return 'Image téléchargée avec succès et convertie en pixels'
    else:
        return 'Aucun fichier sélectionné'


@app.route('/upload/form', methods=['GET'])
def upload_form():
    return render_template("upload_form.html")


def train_image(img_path):
    img = Image.open(img_path)
    img = img.convert('RGB')
    flattened_inputs = np.array(
        list(map(lambda x: [x[0], x[1], x[2]], list(img.getdata())[0:10]))
    ).flatten().astype(np.float32)

    lib.train_classification(flattened_inputs, [1, 1, -1])


if __name__ == "__main__":
    app.run(debug=True)
