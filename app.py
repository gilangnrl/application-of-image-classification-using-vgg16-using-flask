import os
import time
import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from livereload import Server  # For live reloading during development
from PIL import Image

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Maximum upload size allowed
model_dict = {
            "rps": "./model/model_vgg.h5",
            "corel_10k": "./model/vgg_rgb_corel_10k.hdf5"
        }  # Path to the pre-trained model file

class_list = {
    'rps'       : { 'Paper': 0, 'Rock': 1,'Scissors': 2},
    'corel_10k' : { 'airplane': 0,
                    'african': 1,
                    'f1 car': 2,
                    'bridge': 3,
                    'wind surfing': 4,
                    'view': 5,
                    'aztec': 6,
                    'sailboat': 7,
                    'women': 8,
                    'view art': 9,
                    'furniture': 10,
                    'bear': 11,
                    'ai': 12,
                    'dinosaur': 13,
                    'wall art': 14,
                    'castle': 15,
                    'light house': 16,
                    'city': 17,
                    'statue': 18,
                    'drink': 19,
                    'food': 20,
                    'martials art': 21,
                    'deer': 22,
                    'space': 23,
                    'greek': 24,
                    'halloween': 25,
                    'old airplane': 26,
                    'air balloon': 27,
                    'bobsled': 28,
                    'bonsai': 29,
                    'bus': 30,
                    'old car': 31,
                    'playing card': 32,
                    'astronaut': 33,
                    'duck': 34,
                    'chinaware': 35,
                    'doll': 36,
                    'door': 37,
                    'easter egg': 38,
                    'flag': 39,
                    'mask': 40,
                    'agate': 41,
                    'molecule': 42,
                    'atom': 43,
                    'sea nature': 44,
                    'ship': 45,
                    'steam engine': 46,
                    'train': 47,
                    'papuan': 48,
                    'cat': 49,
                    'dog': 50,
                    'flower': 51,
                    'leaf': 52,
                    'fungus': 53,
                    'cave': 54,
                    'plant': 55,
                    'autumn': 56,
                    'cloud': 57,
                    'firework': 58,
                    'tree': 59,
                    'arctic': 60,
                    'interior': 61,
                    'glacier': 62,
                    'city night': 63,
                    'monument valley': 64,
                    'garden': 65,
                    'swimming': 66,
                    'sunset': 67,
                    'waterfall': 68,
                    'water wave': 69,
                    'skiing': 70,
                    'liquid art': 71,
                    'abstract art 1': 72,
                    'fractal art': 73,
                    'frost': 74,
                    'abstract art 2': 75,
                    'roman art': 76,
                    'egypt': 77,
                    'butterfly': 78,
                    'bob cat': 79,
                    'cougar': 80,
                    'antelope': 81,
                    'eagle': 82,
                    'elephant': 83,
                    'fish': 84,
                    'coyote': 85,
                    'bighorn sheep': 86,
                    'horse': 87,
                    'steam locomotive': 88,
                    'cheetah': 89,
                    'lion': 90,
                    'reptile': 91,
                    'bird': 92,
                    'owl': 93,
                    'dolphin': 94,
                    'monkey': 95,
                    'hippo': 96,
                    'tiger': 97,
                    'wolf': 98,
                    'polar bear': 99
 }
}  # Example class labels


# Function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to load and preprocess the uploaded image
def load_image(file):
    file.save(os.path.join('static', 'temp.jpg'))
    img = np.array(Image.open(file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, (224, 224)).astype('float32') / 255, axis=0)
    return img

# Function to render the prediction result using a template
def predict_result(model_name, run_time, probs, img):
    idx_pred = probs.index(max(probs))
    labels = list(class_list[model_name].keys())
    return render_template('/result_select.html', labels=labels,
                           probs=probs, model=model_name, pred=idx_pred,
                           run_time=run_time, img=img)

# Padds headers to the server response instructing browsers not to cache the content, 
# ensuring the retrieval of fresh data from the server for each request.
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

# Route to the main Main page in index.html
@app.route("/")
def index():
    return render_template('/index.html')

# Route to handle image upload and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files["file"]
    model_name = request.form["model_name"]
    if file and allowed_file(file.filename):
        if model_name in model_dict:
            model = load_model(model_dict[model_name])
        else:
            model = load_model(model_dict[0])
        img = load_image(file)
        start = time.time()
        pred = model.predict(img)[0]
        labels = (pred > 0.5).astype(int)  # Example thresholding for binary classification
        print(labels)
        runtimes = round(time.time() - start, 4)
        respon_model = [round(elem * 100, 2) for elem in pred]
        return predict_result(model_name, runtimes, respon_model, 'temp.jpg')
    else:
        return render_template('/invalid.html')

# Run the server with live reload functionality during development
if __name__ == "__main__":
    server = Server(app.wsgi_app)
    server.watch("templates/*.*")  # Watch for changes in template files
    server.serve()
