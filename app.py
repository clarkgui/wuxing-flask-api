import socket

# 保存原始 getfqdn 函数
_real_getfqdn = socket.getfqdn

# 替换 getfqdn，防止中文主机名引发 UnicodeDecodeError
def safe_getfqdn(name=None):
    try:
        return _real_getfqdn(name)
    except UnicodeDecodeError:
        return 'localhost'

socket.getfqdn = safe_getfqdn

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import traceback

from predict import extract_facial_features, load_five_element_model, predict_five_element


app = Flask(__name__)
CORS(app)  # 允许跨域请求（供小程序调用）

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型（只加载一次）
model, label_encoder, feature_columns, _ = load_five_element_model()

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 提取特征
        facial_indices = extract_facial_features(file_path)
        # 预测体质
        result = predict_five_element(model, label_encoder, feature_columns, facial_indices)
        # 组合返回值
        response = {
            'prediction': result,
            'features': facial_indices
        }
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)


