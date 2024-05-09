import sys
from pathlib import Path
import os
import uuid
import cv2
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify, session
from modules.image_processing import init_sam_model, process_image

import numpy as np
import modules.utils as utils
from werkzeug.utils import secure_filename

app = Flask(__name__)
#SECRET_KEY ensure every time run it, it's clean and new data
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or str(uuid.uuid4())
mask_utils = init_sam_model()

@app.route('/')
def index():
    return render_template('index.html')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['STATIC_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Add this function to your app.py
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/static/<path:filename>')
def send_static_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        utils.check_folder(app.config['UPLOAD_FOLDER'])
        file.save(path)
        session['uploaded_image'] = filename
        return redirect(url_for('index'))
    return redirect(request.url)

@app.route('/process_points', methods=['POST'])
def process_points():
    if not 'uploaded_image' in session:
        session['uploaded_image'] = 'default.png'
    
    data = request.get_json(force=True)
    points = data['points']
    displayed_size = data['img_size']

    if len(points) < 1:
        return jsonify({'status': 'error', 'message': 'No points selected'})

    # 处理图像
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_image'])

    original_size = utils.get_image_size(image_path)

    #because the image is resized to fit the screen, we need to rescale the points
    points = utils.rescale_points(points, original_size, displayed_size)
    point_array = []
    label_array = []
    for point in points:
        point_array.append([point['x'], point['y']])
        label_array.append(1)
    
    input_point = np.array(point_array)
    input_label = np.array(label_array)

    file_name = session['uploaded_image'].split('.')[0]
    save_path = os.path.join(app.config['OUTPUT_FOLDER'], file_name)
    save_path = utils.get_new_path_if_exist(save_path)
    utils.check_folder(save_path)
    output_image_path, mask_image_path = process_image(image_path, mask_utils, input_point, input_label, save_path)

    # / use in unix, \ use in windows, it's so annoying
    output_image_path = output_image_path.replace('\\', '/')
    
    output_image_path = output_image_path.split('output/')[1]

    
    return jsonify({
        'output_image_path': output_image_path,
        "mask_image_path": mask_image_path
    })



@app.route('/get_auto_masks', methods=['POST'])
def get_auto_masks():
    if not 'uploaded_image' in session:
        session['uploaded_image'] = 'default.png'
    # 处理图像
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], session['uploaded_image'])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = mask_utils.get_auto_masks(image)
    print(f">>>>>>{jsonify(res)}")

    return jsonify(res)

@app.route('/processed_image/<path:image_path>')
def processed_image(image_path):
    return send_from_directory(app.config['OUTPUT_FOLDER'], image_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8079)
