from flask import Flask, render_template, request

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
        return 'Image téléchargée avec succès'
    else:
        return 'Aucun fichier sélectionné'

@app.route('/upload/form', methods=['GET'])
def upload_form():
    return render_template("upload_form.html")

if __name__ == "__main__":
    app.run(debug=True)
