import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_five_element_model(excel_path):
    """从Excel训练并保存五行分类模型（使用全部样本）"""
    # 1. 读取Excel数据
    df = pd.read_excel(excel_path)

    # 2. 检查数据格式
    required_features = [
        "额骨大小宽比例", "颧下颌宽指数", "颧额宽指数Ⅰ", "颧额宽指数Ⅱ",
        "形态面指数", "口宽指数", "容貌面指数", "唇指数",
        "高点全唇高指数", "下巴长指数"
    ]

    # 确保所有必需特征存在
    missing_features = [feat for feat in required_features if feat not in df.columns]
    if missing_features:
        raise ValueError(f"数据集中缺少必需特征: {missing_features}")

    # 3. 分离特征和标签
    X = df[required_features]
    y = df["五行判断"]  # 确保标签列名为"五行判断"

    # 4. 编码标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # 5. 使用全部数据训练（不再划分测试集）
    # 直接使用整个数据集作为训练集
    X_train = X
    y_train = y_encoded

    # 6. 训练随机森林模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 7. 计算并显示训练集准确率（可选）
    train_accuracy = model.score(X_train, y_train)
    print(f"训练集准确率: {train_accuracy:.4f}")

    # 8. 保存完整模型包
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_columns': required_features,
        'accuracy': train_accuracy  # 保存训练集准确率
    }
    joblib.dump(model_data, "five_element_rf_model_full.joblib")
    print("模型已保存为 five_element_rf_model_full.joblib")


# 使用示例
if __name__ == "__main__":
    excel_path = "D:\project\wuxing\人脸识别参数及五行判断（0-239）（纯净版）.xlsx"  # 替换为实际Excel路径
    train_five_element_model(excel_path)