from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow
# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#for hashtags
import instaloader

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_resnet50.h5'

# Load your trained model
model = load_model(MODEL_PATH)


def ins(x):
    l = instaloader.Instaloader()
    gg = instaloader.TopSearchResults(l.context,x)
    ll = []
    for x in gg.get_hashtag_strings():
        ll.append(x)
    return ll



def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)


    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

   

    preds = model.predict(x)
    pred=np.argmax(preds, axis=1)
    result="car"
    if pred==0:
        result="Audi"
    elif pred==1:
        result="Lamborghini"
    else:
        result="Mercedes"
    
    
    return ins(result)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname('uploads/')
        file_path = os.path.join(
            basepath,  secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        text = ""
        for tag in preds:
            text = text + '  #' + str(tag)
        text = ' ' + text
        return text
    return None


if __name__ == '__main__':
    app.run(debug=True)
