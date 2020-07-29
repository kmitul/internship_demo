from flask import Flask, render_template, request
from werkzeug import secure_filename
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded/image'

@app.route('/demo')
def upload_f():
    return render_template('base.html')

@app.route('/')
def index():
    return render_template('index.html')

def getxraylabel():
    labels = ['NORMAL', 'PNEUMONIA']
    for name in os.listdir('uploaded/image/'):
      path = 'uploaded/image/' + name
      img = image.load_img(path, target_size=(300,300))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis =0)

      images = np.vstack([x])
      classes = xray_model.predict(images, batch_size = 10)
      print(classes)

      return labels[int(classes[0][0])]

def gettumorlabel():
    labels = ['MENINGIOMA', 'GLIOMA', 'PITUITARY ADENOMA']
    for name in os.listdir('uploaded/image/'):
      path = 'uploaded/image/' + name
      img = image.load_img(path, color_mode='grayscale', target_size=(512, 512, 1))
      x = image.img_to_array(img)
      x = x/255
      x = np.repeat(x, 3, 2)

      x = np.expand_dims(x, axis =0)
      images = np.vstack([x])
      classes = tumor_model.predict(images, batch_size = 10)

      print(classes)

      return labels[np.argmax(classes[0])]

@app.route('/demo/xray', methods = ['GET','POST'])
def upload_xrayfile():
    if request.method == 'POST':
        f = request.files['file']
        img = f.filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        label = getxraylabel()
        os.remove('uploaded/image/'+f.filename)
        return render_template('pred.html', label = label, img = img)

    if request.method == 'GET':
        return render_template('pred.html')

@app.route('/demo/tumor', methods = ['GET','POST'])
def upload_tumorfile():
    if request.method == 'POST':
        f = request.files['file']
        img = f.filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        label = gettumorlabel()
        os.remove('uploaded/image/'+f.filename)
        return render_template('pred.html', label = label, img=img)

    if request.method == 'GET':
        return render_template('pred.html')

if __name__ == '__main__':
    #loading all the models
    xray_model = tf.keras.models.load_model('models/chest-xray-model.h5')
    tumor_model = tf.keras.models.load_model('models/classification_trial1__hdf5.h5')
    app.run()
