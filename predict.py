import cv2
import dlib
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 初始化dlib人脸检测器和特征点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"D:\project\wuxing\sl python\shape_predictor_81_face_landmarks.dat")


def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(np.array(p2) - np.array(p1))


def get_facial_indices(shape):
    """计算所有面部指数"""
    points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(81)])

    # 基础测量指标
    forehead_min = calculate_distance(points[68], points[73])  # 额骨最小宽（头角间距）
    forehead_max = calculate_distance(points[77], points[78])  # 额骨最大宽（太阳穴间距）
    zygomatic_width = calculate_distance(points[1], points[15])  # 颧骨宽度
    jaw_width = calculate_distance(points[3], points[13])  # 下颌宽度
    face_width = calculate_distance(points[1], points[15])  # 面宽（颧点间距）
    mouth_width = calculate_distance(points[54], points[48])  # 口裂宽
    morph_face_height = calculate_distance(points[8], points[27])  # 形态面高（鼻根到下巴）
    app_face_height = calculate_distance(points[71], points[8])  # 容貌面高（神庭到下巴）
    lip_high_height = calculate_distance(points[50], points[58])  # 高点全唇红厚
    lip_height = calculate_distance(points[51], points[57])  # 唇高
    xiaba_length = calculate_distance(points[33], points[8])  # 下巴长度

    # 计算面型指数
    indices = {
        "额骨大小宽比例": forehead_min / forehead_max,
        "颧下颌宽指数": jaw_width / zygomatic_width,
        "颧额宽指数Ⅰ": forehead_min / zygomatic_width,
        "颧额宽指数Ⅱ": forehead_max / zygomatic_width,
        "形态面指数": morph_face_height / face_width,
        "口宽指数": mouth_width / jaw_width,
        "容貌面指数": app_face_height / face_width,
        "唇指数": lip_height / mouth_width,
        "高点全唇高指数": lip_high_height / mouth_width,
        "下巴长指数": xiaba_length / app_face_height
    }
    return indices


def extract_facial_features(image_path, visualize=False):
    """从图片中提取面部特征"""
    # 支持中文路径读取
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        raise ValueError("未检测到人脸")

    # 只处理第一张人脸
    face = faces[0]
    shape = predictor(gray, face)

    if visualize:
        img_vis = img.copy()
        # 绘制所有特征点及序号
        for i in range(81):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(img_vis, (x, y), 2, (0, 255, 0), -1)
        # 显示带标记的灰度图
        cv2.imshow("Landmarks Visualization", img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 计算指数
    return get_facial_indices(shape)

#---------------------------------------------------------------------------------------------

def load_five_element_model(model_path=r"D:\project\wuxing\sl python\five_element_rf_model_full.joblib"):
    """加载五行预测模型"""
    model_data = joblib.load(model_path)
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    feature_columns = model_data['feature_columns']

    # 获取特征重要性
    feature_importances = model.feature_importances_*100
    return model, label_encoder, feature_columns, feature_importances

#-----------------------------------------------------------------------------------------------

def predict_five_element(model, label_encoder, feature_columns, facial_indices):
    """预测五行属性"""
    # 确保特征顺序与模型训练时一致
    features = np.array([facial_indices[col] for col in feature_columns]).reshape(1, -1)
    prediction = model.predict(features)
    return label_encoder.inverse_transform(prediction)[0]


# 使用示例
if __name__ == "__main__":
    # 加载预训练模型
    model, label_encoder, feature_columns, feature_importance = load_five_element_model()

    # 输入图片路径
    image_path = r"D:\project\wuxing\backend\7032d2e765f1d823ec441a2d33225b7.png"  # 替换为实际图片路径

    try:
        # 提取面部特征
        facial_indices = extract_facial_features(image_path, visualize=True)

        # 打印提取的特征
        print("\n===== 提取的面部特征 =====")
        for feature, value in facial_indices.items():
            print(f"{feature}: {value:.6f}")

        # 预测五行属性
        prediction = predict_five_element(model, label_encoder, feature_columns, facial_indices)
        print(f"\n预测结果: 此人属于 {prediction} 型面相")

        # 打印特征重要性
        print("\n===== 特征重要性排序 =====")
        # 创建特征名和重要性的对应列表
        importance_list = list(zip(feature_columns, feature_importance))
        # 按重要性值降序排序
        importance_list.sort(key=lambda x: x[1], reverse=True)

        # 打印排序后的特征重要性
        for feature, importance in importance_list:
            print(f"{feature}: {importance:.4f}")


    except Exception as e:
        print(f"处理出错: {str(e)}")