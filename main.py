from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug import secure_filename
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded/image'
app._static_folder = 'static'

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


#custom objects for chandan's model
from tensorflow.keras import backend as K

epsilon=1e-6
def dice_coeff(y_pred,y_true):
    y_predf=K.flatten(y_pred)
    y_truef=K.flatten(y_true)
    numerator= 2*K.sum(y_predf*y_truef)+epsilon
    denominator=K.sum(y_predf)+K.sum(y_truef) +epsilon
    return numerator/denominator

def dice_coeff_loss(y_pred,y_true):
    return -dice_coeff(y_pred,y_true)


def getseglabel():

    for name in os.listdir('uploaded/image/'):
        path = 'uploaded/image/' + name
        img = cv2.imread(path)
        img = cv2.resize(img ,(256, 256))
        img = img / 255
        img = img[np.newaxis, :, :, :]
        pred = seg_model.predict(img)
        print('debug1')
        return pred


@app.route('/demo/seg', methods = ['GET','POST'])
def upload_segfile():
    if request.method == 'POST':
        f = request.files['file']
        img = f.filename
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        fig = getseglabel()
        #changing extension .tiff to .jpg
        imgjpg = img.split('.')[0] + '.jpg'
        print(imgjpg)
        plt.imsave(os.path.join('static/',imgjpg), np.squeeze(fig))
        os.remove('uploaded/image/'+f.filename)
        return render_template('seg.html', img= imgjpg)

    if request.method == 'GET':
        return render_template('seg.html')


@app.route('/demo/<path:filename>')
def send_file(filename):
    return send_from_directory('static', filename)


if __name__ == '__main__':
    #loading all the models
    xray_model = tf.keras.models.load_model('models/chest-xray-model.h5')
    tumor_model = tf.keras.models.load_model('models/classification_trial1__hdf5.h5')
    seg_model = tf.keras.models.load_model('models/unet_brain_mri_seg.hdf5', custom_objects={'dice_coeff_loss': dice_coeff_loss, 'iou':tf.keras.metrics.MeanIoU(num_classes=2), 'dice_coeff': dice_coeff})
    app.run()
