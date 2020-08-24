
#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Import Keras dependencies
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.keras.preprocessing import image

# Import other dependecies
import numpy as np
import h5py
from PIL import Image
import PIL
import os

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
MODEL_ARCHITECTURE = './model/model.json'
MODEL_WEIGHTS = './model/model_100.h5'

# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')


# ::: MODEL FUNCTIONS :::
def model_predict(img_path, model):
	'''
		Args:
			-- img_path : an URL path where a given image is stored.
			-- model : a given Keras CNN model.
	'''

	IMG = image.load_img(img_path, target_size=(224,224))
	print(type(IMG))

	# Pre-processing the image
	IMG_ = image.img_to_array(IMG)
	print(IMG_.shape)
	#IMG_ = np.true_divide(IMG_, 255)
	#IMG_ = IMG_.reshape(1, 224, 224, 3)
	#print(type(IMG_), IMG_.shape)
	IMG_ = np.expand_dims(IMG_, axis=0)
	IMG_ = preprocess_input(IMG_, mode='caffe')

	print(model)

	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
	prediction = model.predict(IMG_)

	return prediction


# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():

	# Constants:
	classes = {'TRAIN': ['NSFW', 'SFW'],
	           'VALIDATION': ['NSFW', 'SFW'],
	           'TEST': ['NSFW', 'SFW']}

	if request.method == 'POST':

		# Get the file from post request
		f = request.files['file']

		# Save the file to ./uploads
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(
			basepath, 'uploads', secure_filename(f.filename))
		f.save(file_path)

		# Make a prediction
		preds = model_predict(file_path, model)
		pred_class = decode_predictions(preds, top=1)
		result = str(pred_class[0][0][1])
		return result
	return None


if __name__ == '__main__':
	app.run(debug = True)
