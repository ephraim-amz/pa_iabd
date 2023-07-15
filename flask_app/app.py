from flask import Flask, render_template, request, flash
from PIL import Image
import numpy as np
import linear_classifier
import pmc
import secrets

app = Flask(__name__)

app.config['SECRET_KEY'] = secrets.token_urlsafe(5)

@app.route('/')
def index():
    return render_template("index.html")


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
        print("Image téléchargée avec succès et convertie en pixels")
    else:
        print("Aucun fichier sélectionné")
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

if __name__ == "__main__":
    app.run(debug=True)
