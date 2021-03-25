import os
from uuid import uuid4

import cv2
from flask import Flask, request, render_template, send_from_directory


app = Flask(__name__)
# app = Flask(__name__, static_folder="images")


APP_ROOT = os.path.dirname(os.path.abspath(__file__))

classes = ['Fresh Banana', 'Fresh Blueberry', 'Fresh Huckleberry',
           'Fresh Orange', 'Rotten Banana', 'Rotten Blueberry',
           'Rotten Orange']
# la je reprends le serveur keras et jessaye dintegré la camera dedans directement a voir si cest le mieu afaire
cap = cv2.VideoCapture(0)
# je me dit dans un premier temps extraire et enregistrer des image depuis la camera live voir si le serveur fonctionne puis essayé de passé directement par la camera sans interface graphique
# ce serai chiant de tt reecrire les xml deja les label ca va je les ai ce serai cool de convertir ou de trouvé une fonction qui me pond le xml des data ca mevite un travail impossible
# sinon si je doit cree lencadrement des fruit avec open cv les mask pour reconnaitre donc une nouvelle ia qui risque de me prendre du temps faut prendre la decision rapidement
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        # import tensorflow as tf
        import numpy as np
        from keras.preprocessing import image

        from keras.models import load_model
        new_model = load_model('model.h5')
        new_model.summary()
        test_image = image.load_img('images\\' + filename, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = new_model.predict(test_image)
        result1 = result[0]
        for i in range(6):

            if result1[i] == 1.:
                break;
        prediction = classes[i]

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("template.html", image_name=filename, text=prediction)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    app.run(debug=False)
