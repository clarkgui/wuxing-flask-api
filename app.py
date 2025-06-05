from flask import Flask, request, jsonify
import joblib
from predict import predict_five_element
import os

app = Flask(__name__)

# 加载模型和特征信息
model = joblib.load('model/random_forest_model.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')
feature_columns = joblib.load('model/feature_columns.pkl')

@app.route('/')
def index():
    return 'Flask API is running.'

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'Missing JSON data'}), 400

        # 调用预测函数
        result = predict_five_element(model, label_encoder, feature_columns, json_data)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)



