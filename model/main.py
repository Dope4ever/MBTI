import joblib
from sklearn.preprocessing import StandardScaler


def validate_scaler():
    try:
        objects = joblib.load("feature_processing_objects.pkl")
        scaler = objects['meta_scaler']

        print("Scaler验证结果:")
        print(f"类型: {type(scaler)}")
        print(f"均值: {scaler.mean_}")
        print(f"标准差: {scaler.scale_}")
        print(f"样本数: {scaler.n_samples_seen_}")

        assert isinstance(scaler, StandardScaler), "对象类型错误"
        assert hasattr(scaler, 'mean_'), "缺少mean_属性"
        assert scaler.mean_.shape == (5,), "维度不正确"

        print("✅ Scaler验证通过")
        return True
    except Exception as e:
        print(f"❌ 验证失败: {str(e)}")
        return False


if __name__ == "__main__":
    validate_scaler()