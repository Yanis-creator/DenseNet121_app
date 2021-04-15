import os
import cv2
import sys
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from flask import send_file


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model 
import matplotlib.image as mpimg
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
from util import base64_to_pil

app = Flask(__name__)



model = load_model('DenseNet.h5')
img_size = (224, 224)
last_conv_layer_name = "relu"
classifier_layer_names = [
        "avg_pool",
        "predictions",
 ]

def preprocess_the_input(img):
    img = img.resize(img_size)
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def model_predict(img, model):
    img = img.resize(img_size)
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

def cam(img):
    img_array = preprocess_the_input(img)
    img = keras.preprocessing.image.img_to_array(img)


    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

        # Second, we create a model that maps the activations of the last conv
        # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
            # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
            # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)


    heatmap = np.uint8(255 * heatmap)

        # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

        # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

        # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
    save_path = "img_cam.jpg"
    superimposed_img.save(save_path)
    return None

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

    
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        img1 = base64_to_pil(request.json)
        #cam(img1)
        preds = model_predict(img, model)
        # Process your result for human
        pred_class =  decode_predictions(preds)  # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        result = result.replace('_', ' ').capitalize()
        return jsonify(result=result) #, probability=pred_proba)
    return None
@app.route('/analysis')
def out():
    save_path = 'img_cam.jpg'
    return send_file(save_path, mimetype='image/jpg')



if __name__ == "__main__":
    app.run(port = 8080, debug =True)
