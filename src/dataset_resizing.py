from PIL import Image
import os
import shutil

# Chemin vers le dossier contenant les images
type_terrain = "basket"
dossier_images = f"./img/{type_terrain}"

# Chemin vers le dossier de sortie pour les images redimensionnées
dossier_sortie = f"./img/{type_terrain}_resize"

# Vérifier si le dossier de sortie existe, sinon le créer
if not os.path.exists(dossier_sortie):
    os.makedirs(dossier_sortie)

# Parcours des fichiers dans le dossier
for nom_fichier in os.listdir(dossier_images):
    chemin_image = os.path.join(dossier_images, nom_fichier)

    # Vérifier la taille de l'image
    image = Image.open(chemin_image)
    largeur_image, hauteur_image = image.size

    if largeur_image > 300 or hauteur_image > 300:
        # Redimensionner l'image si la largeur ou la hauteur est supérieure à 300 pixels
        largeur_cible = 300
        facteur_redimensionnement = min(1.0, float(largeur_cible) / largeur_image, float(largeur_cible) / hauteur_image)
        nouvelle_largeur = int(largeur_image * facteur_redimensionnement)
        nouvelle_hauteur = int(hauteur_image * facteur_redimensionnement)
        image_redimensionnee = image.resize((nouvelle_largeur, nouvelle_hauteur), Image.BILINEAR)

        # Créer un nouveau nom de fichier pour l'image redimensionnée
        nom_fichier_redimensionne = f"redimensionne_{nom_fichier}"
        chemin_sortie = os.path.join(dossier_sortie, nom_fichier_redimensionne)

        # Convertir l'image en mode RVB avant de la sauvegarder en JPEG
        image_redimensionnee = image_redimensionnee.convert("RGB")

        # Sauvegarder l'image redimensionnée en JPEG
        image_redimensionnee.save(chemin_sortie, "JPEG")

    else:
        # Copier les images d'origine dans le dossier de sortie sans les redimensionner
        chemin_sortie = os.path.join(dossier_sortie, nom_fichier)
        shutil.copy2(chemin_image, chemin_sortie)
