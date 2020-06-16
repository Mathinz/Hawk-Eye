
# Import libraries
from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from werkzeug.utils import secure_filename

import warnings
warnings.filterwarnings("ignore")
import os
import sys
import imghdr
import cv2
import skimage.io
import logging as log
from time import time, sleep
from datetime import datetime
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import shutil
import pandas as pd
import trace

# from openvino.inference_engine import IENetwork, IEPlugin
from camera import VideoCamera
import threading

from tensorflow import keras
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)
app.config['SECRET_KEY'] = 'cpsthrtdtct'

config_p = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)
k_session = tf.compat.v1.Session(config=config_p)

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import visualize
import mrcnn.model as modellib

from cps_utils import save_captured_image

# Import config
from cps_maskrcnn import cpsConfig
config = cpsConfig()
# config.display()

# Directories
MODEL_DIR = os.path.join(ROOT_DIR, "logs")  # logs
# Path to pre-trained weights
CPS_WEIGHTS_PATH = "weights/"

# Class names
# Index of the class in the list is its ID
class_names = ['BG', 'mask', 'gun', 'knife', 'car', 'person', 'truck', 'bat', 'motorcycle', 'grenade', 'rpg']

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Set default session
keras.backend.set_session(k_session)

# Load pre-trained weights
CPS_WEIGHTS_FILE = os.path.join(CPS_WEIGHTS_PATH, "mask_rcnn_cps_0012.h5")
model.load_weights(CPS_WEIGHTS_FILE, by_name=True)
model.keras_model._make_predict_function()

@app.route('/')
def index():
    session.clear()
    return render_template('index.html')


def run_detect(filename):

    # Convert png with alpha channel with shape[2] == 4 into shape[2] ==3 RGB images
    image = skimage.io.imread(filename)
    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Run detection
    with k_session.as_default():
        with k_session.graph.as_default():
            tic = time()
            results = model.detect([image], verbose=1)
            toc = time()
            r = results[0]
            inf_time = toc - tic

    # Just apply mask then save images
    captured = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                           class_names, r['scores'],
                                           None, None, None, True, True, None, None)

    if captured:
        saved_file_name = 'static/masked/temp_masked.png'

    return True, saved_file_name, inf_time

def rename_file(file_path, saved_file_path):
    bn = os.path.basename(file_path)
    os.rename(saved_file_path, 'static/masked/masked_' + bn)
    session['result_img'] = bn
    return bn


# Model paths for use with the stick
model_xml = "IR_model/IR26_model.xml"
model_bin = "IR_model/IR26_model.bin"

def detect_using_stick(im_path):
    session.clear()
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

    # Plugin initialization for specified device
    plugin = IEPlugin(device="MYRIAD")

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    # Prepare image
    n, c, h, w  = net.inputs[input_blob].shape
    print(n, h, w, c)
    #prepimg = np.ndarray(shape=(n, c, h, w))

    # Read image as grayscale
    im = cv2.imread(im_path)

    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = np.copy(im)
    image = cv2.resize(image, (200, 200))
    image = image.transpose((2,0,1))
    p_image = image.reshape(1, 3, 200, 200)

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = plugin.load(network=net)
    del net

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(1))
    infer_time = []
    t0 = time()
    tic = time()
    res = exec_net.infer(inputs={input_blob: p_image})
    toc = time()
    inf_time = toc-tic
    infer_time.append((time()-t0)*1000)
    log.info("Average running time of one iteration: {} ms".format(np.average(np.asarray(infer_time))))

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]

    cl = get_class(res)

    del exec_net
    del plugin

    return cl, inf_time

def get_class(res):
    if res[0][3] == 1:
        return 'Pistol'
    elif res[0][4] == 1:
        return 'RPG'
    elif res[0][0] == 1:
        return 'Grenade'
    elif res[0][1] == 1:
        return 'Machine Gun'
    elif res[0][2] == 1:
        return 'Masked face'
    else:
        return 'Unknown output'

############################################################
#                   MASK RCNN ROUTES
############################################################
@app.route('/mrcnn/upload_image', methods=['GET', 'POST'])
def mrcnn_upload_image():
    session.clear()
    # Validation images extension
    image_type_ok_list = ['jpeg', 'png', 'gif', 'bmp']
    model_name = 'MaskRCNN'
    if request.method == 'POST':
        if 'file' not in request.files:
            response = {
                'message': "No file uploaded within the POST body."
            }
            return jsonify(response), 400

        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)

        session['image_path'] = file_path

        image_type = imghdr.what(file_path)

        if image_type in image_type_ok_list:
            result, saved_file_path, inf_time = run_detect(file_path)
            result_img = rename_file(file_path, saved_file_path)
            session['inf_time'] = inf_time

        return redirect(url_for('display_mrcnn_result'))
    return render_template('img_upload.html', model_name=model_name)

@app.route('/mrcnn/capture_image', methods=['GET', 'POST'])
def mrcnn_capture_image():
    session.clear()
    image_type_ok_list = ['jpeg', 'png', 'gif', 'bmp']
    model_name = 'MaskRCNN'
    if request.method == 'POST':
        file = request.form['file']
        image_name = str(request.form['image_name'])
        image_path = save_captured_image(file, image_name)
        image_type = imghdr.what(image_path)
        if image_type in image_type_ok_list:
            result, saved_file_path, inf_time = run_detect(image_path)
            result_img = rename_file(image_path, saved_file_path)
            session['inf_time'] = inf_time

            return redirect(url_for('display_mrcnn_result'))
        # return render_template('result_mrcnn.html')
    return render_template('img_capture.html', model_name=model_name)

@app.route('/mrcnn/result')
def display_mrcnn_result():
    return render_template('result_mrcnn.html')

############################################################
#                       CNN ROUTES
############################################################
class_dict = {'Grenade': 0, 'Machine Guns': 1, 'Masked Face': 2, 'Pistol': 3, 'RPG': 4,
 'Motorcycle': 5, 'Car': 6, 'Knife': 7, 'Bat': 8, 'Truck': 9}

model_loaded = load_model('weights/best_model16.h5')

def get_key(val):
    for key, value in class_dict.items():
         if val == value:
             return key

def run_cnn_inference(image_path):
    image = Image.open(image_path)
    with k_session.as_default():
        with k_session.graph.as_default():
            tic = time()
            prediction = model_loaded.predict(np.expand_dims(image.resize((300,300)),axis=0)/255)
            toc = time()
    inf_time = toc - tic
    label = get_key(np.argmax(prediction))
    confidence = np.max(prediction)*100

    return label, confidence, inf_time

@app.route('/cnn/upload_image', methods=['GET', 'POST'])
def cnn_upload_image():
    session.clear()
    # Validation images extension
    image_type_ok_list = ['jpeg', 'png', 'gif', 'bmp']
    model_name = 'CNN'
    if request.method == 'POST':
        if 'file' not in request.files:
            response = {
                'message': "No file uploaded within the POST body."
            }
            return jsonify(response), 400
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)

        image_type = imghdr.what(file_path)

        if image_type in image_type_ok_list:
            pred_class,confidence, inf_time = run_cnn_inference(file_path)
            image_name = os.path.basename(repr(file_path))
            image_name = image_name[:-1]
            session['image_name'] = image_name
            session['pred_class'] = pred_class
            session['confidence'] = confidence
            session['inf_time'] = inf_time
            return redirect(url_for('display_cnn_result'))

    return render_template('img_upload.html', model_name=model_name)

@app.route('/cnn/capture_image', methods=['GET', 'POST'])
def cnn_capture_image():
    session.clear()
    image_type_ok_list = ['jpeg', 'png', 'gif', 'bmp']
    model_name = 'CNN'
    is_captured = False
    if request.method == 'POST':
        file = request.form['file']
        image_name = str(request.form['image_name'])
        image_path = save_captured_image(file, image_name)
        image_type = imghdr.what(image_path)
        if image_type in image_type_ok_list:
            pred_class, confidence, inf_time = run_cnn_inference(image_path)
            session['pred_class'] = pred_class
            session['confidence'] = confidence
            session['inf_time'] = inf_time
            session['cap_image_name'] = os.path.basename(image_path)

            return redirect(url_for('display_cnn_result'))
    return render_template('img_capture.html', model_name=model_name)

@app.route('/cnn/result')
def display_cnn_result():
    return render_template('result_cnn.html')

############################################################
#                       IR model ROUTES
############################################################
@app.route('/IR/upload_image', methods=['GET', 'POST'])
def IR_upload_image():
    session.clear()
    # Validation images extension
    image_type_ok_list = ['jpeg', 'png', 'gif', 'bmp']
    model_name = 'IR Model'
    if request.method == 'POST':
        if 'file' not in request.files:
            response = {
                'message': "No file uploaded within the POST body."
            }
            return jsonify(response), 400
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static/uploads', secure_filename(f.filename))
        f.save(file_path)

        image_type = imghdr.what(file_path)

        if image_type in image_type_ok_list:
            pred_class, inf_time = detect_using_stick(file_path)
            session['pred_class'] = pred_class
            image_name = os.path.basename(repr(file_path))
            image_name = image_name[:-1]
            session['image_name'] = image_name
            session['inf_time'] = inf_time

            return redirect(url_for('display_IR_result'))
    return render_template('img_upload.html', model_name=model_name)

@app.route('/IR/capture_image', methods=['GET', 'POST'])
def IR_capture_image():
    session.clear()
    image_type_ok_list = ['jpeg', 'png', 'gif', 'bmp']
    model_name = 'IR Model'
    is_captured = False
    if request.method == 'POST':
        file = request.form['file']
        image_name = str(request.form['image_name'])
        image_path = save_captured_image(file, image_name)
        image_type = imghdr.what(image_path)
        if image_type in image_type_ok_list:
            pred_class, inf_time = detect_using_stick(image_path)
            session['pred_class'] = pred_class
            session['inf_time'] = inf_time
            session['cap_image_name'] = os.path.basename(image_path)

            return redirect(url_for('display_IR_result'))
    return render_template('img_capture.html', model_name=model_name)

@app.route('/IR/result')
def display_IR_result():
    return render_template('result_IR.html')

############################################################
#                       Motion Detector ROUTES
############################################################
@app.route('/motion_detector', methods=['GET', 'POST'])
def motion_detector():
    qs = request.args.get('model')
    session['dt'] = datetime.now()
    if request.method == 'POST':
        return redirect(url_for('detect_motion', model=qs))
    return render_template('motion_detector.html', qs=qs)

def video_stream():
    video_camera = VideoCamera()
    video_camera.start_motion_detector()

@app.route('/motion_detector/open_camera')
def detect_motion():
    if os.path.isdir('static/motion_images'):
        shutil.rmtree('static/motion_images')
    if not(os.path.isdir('static/motion_images')): # if directory for motion-images doesn't exist then create it.
        os.makedirs('static/motion_images')
    
    threading.Thread(target=video_stream()).start()

    qs = request.args.get('model')
    return redirect(url_for('images_grid', model=qs))

@app.route('/motion_detector/motion_images', methods=['GET', 'POST'])
def images_grid():
    results = {}
    images_count = len(os.listdir('static/motion_images'))
    if request.method == 'POST':
        if request.form['model'] == 'cnn':
            images_list = request.form.getlist('motion-image')
            for image in images_list:
                results[image] = run_cnn_inference('static/motion_images/'+image+'.png')
            return render_template('motion_images_result.html', model=request.form['model'], results=results)
        elif request.form['model'] == 'ir':
            images_list = request.form.getlist('motion-image')
            for image in images_list:
                results[image] = detect_using_stick('static/motion_images/'+image+'.png')
            return render_template('motion_images_result.html', model=request.form['model'], results=results)
        elif request.form['model'] == 'mrcnn':
            images_list = request.form.getlist('motion-image')
            for image in images_list:
                # store the inference time on index 1
                _, saved_file_path, inf_time = run_detect('static/motion_images/'+image+'.png')
                results[image] = [inf_time] # store inf time at index 0
                results[image].append(rename_file('static/motion_images/'+image+'.png', saved_file_path)) # store masked image at index 1
                print(results)
            return render_template('motion_images_result.html', model=request.form['model'], results=results)
    return render_template('motion_images_grid.html', images_count=images_count)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
