# 导入必要的库
import re  # 正则表达式库
import joblib  # 用于加载和保存模型
import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
# 导入CatBoost模型
from catboost import CatBoostClassifier
from nltk.stem import WordNetLemmatizer  # NLTK库中的词形还原器
from scipy.sparse import hstack  # 用于合并稀疏矩阵
import nltk  # 自然语言处理库
from textblob import TextBlob  # 用于情感分析

# 添加预处理函数（需与训练代码保持一致）
def clean_text(text):
    """文本清洗函数"""
    text = text.lower()  # 转为小写
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = re.sub(r'\s+', ' ', text)  # 合并多个空格
    return text.strip()  # 去掉两端的空格

def add_meta_features(df):
    """添加元特征（与训练代码相同）"""
    # 文本长度特征
    df['post_count'] = df['posts'].apply(len)  # 计算每个帖子中帖子数量
    df['avg_post_length'] = df['posts'].apply(
        lambda x: np.mean([len(p.split()) for p in x]) if len(x) > 0 else 0)  # 计算平均帖子长度
    # 情感分析特征
    df['polarity'] = df['combined_text'].apply(
        lambda x: TextBlob(x).sentiment.polarity)  # 获取文本的情感极性
    df['subjectivity'] = df['combined_text'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity)  # 获取文本的主观性
    # 词汇多样性
    df['lexical_diversity'] = df['combined_text'].apply(
        lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)  # 计算词汇的多样性
    return df  # 返回包含元特征的数据框

# 修改预处理函数
def preprocess_new_data(df, feature_objects):
    """预处理新数据（使用保存的特征处理对象）"""
    # 初始化文本处理工具
    lemma = WordNetLemmatizer()  # 词形还原器
    stop_words = set(nltk.corpus.stopwords.words('english'))  # 停用词集合

    # 加载特征工程对象
    tfidf = feature_objects['tfidf']  # 获取TFIDF特征提取器
    count_vec = feature_objects['count_vec']  # 获取词袋模型
    nmf = feature_objects['nmf']  # 获取NMF模型
    meta_scaler = feature_objects['meta_scaler']  # 获取标准化器

    # 文本预处理（与训练流程完全一致）
    df['combined_text'] = df['posts'].apply(
        lambda x: ' '.join([clean_text(post) for post in x]))  # 清洗文本并合并为一个字段

    # 生成元特征
    df = add_meta_features(df)  # 添加元特征

    # 文本处理管道
    processed_text = df['combined_text'].apply(
        lambda x: ' '.join([lemma.lemmatize(word) for word in x.split()
                            if word not in stop_words and len(word) > 2]))  # 进行词形还原并移除停用词

    # 特征生成
    tfidf_features = tfidf.transform(processed_text)  # 提取TFIDF特征
    bow_features = count_vec.transform(processed_text)  # 提取词袋特征
    nmf_features = nmf.transform(bow_features)  # 使用NMF模型提取主题特征
    meta_features = df[['post_count', 'avg_post_length',
                        'polarity', 'subjectivity', 'lexical_diversity']].values  # 提取元特征

    return hstack([tfidf_features, nmf_features, meta_features])  # 合并所有特征并返回

def predict_mbti_simple(text_df):
    # 加载特征处理管道
    feature_objects = joblib.load('D:/AAAAAAAAAAAAAAAAAAAAAAAgraduate/MBTI/model/feature_processing_objects.pkl')  # 加载预处理对象

    # 生成特征
    X_new = preprocess_new_data(text_df, feature_objects)  # 对新数据进行预处理

    # 定义维度顺序和标签映射
    dim_order = ['E/I', 'S/N', 'T/F', 'J/P']  # MBTI维度顺序
    label_map = {
        'E/I': ('E', 'I'),
        'S/N': ('S', 'N'),
        'T/F': ('T', 'F'),
        'J/P': ('J', 'P')
    }

    # 预测每个维度
    mbti_results = []  # 存储预测结果
    for dim in dim_order:
        # 根据维度选择最佳模型
        if dim == 'E/I':
            model = joblib.load('D:/AAAAAAAAAAAAAAAAAAAAAAAgraduate/MBTI/model/best_model_E_I.pkl')  # SVM
        elif dim == 'S/N':
            model = joblib.load(
                'D:/AAAAAAAAAAAAAAAAAAAAAAAgraduate/MBTI/model/best_model_S_N.pkl')  # Logistic Regression
        elif dim == 'T/F':
            model = joblib.load('D:/AAAAAAAAAAAAAAAAAAAAAAAgraduate/MBTI/model/best_model_T_F.pkl')  # GaussianNB
        elif dim == 'J/P':
            model = joblib.load(
                'D:/AAAAAAAAAAAAAAAAAAAAAAAgraduate/MBTI/model/best_model_J_P.pkl')  # Logistic Regression

        # 确保 X_new 是稠密矩阵
        X_input = X_new.toarray() if hasattr(X_new, 'toarray') else X_new  # 将稀疏矩阵转换为稠密矩阵

        # 获取预测概率
        proba = model.predict_proba(X_input)[:, 1]  # 获取预测的概率
        # 生成类型标签（概率>=50%取第一个标签）
        labels = [label_map[dim][0] if p >= 0.6 else label_map[dim][1] for p in proba]
        mbti_results.append(labels)  # 添加预测结果

    # 组合最终类型
    return pd.DataFrame({
        'MBTI': [
            f"{e}{s}{t}{j}"  # 拼接最终的MBTI类型标签
            for e, s, t, j in zip(*mbti_results)
        ]
    })

def generate_mock_data():
    """生成模拟数据"""
    return pd.DataFrame({
        'posts': [
            # ESTP
            ["I'd also be willing, if needed.",
             "Im rare and limited edition, I guess... Any ideas?"],
            # ENTP
            ["I only rage when i go past my limit",
             "Ohhhh hell yeah. I love to screw with people"],
            # ESFP
            ["Oh my goodness! Is this still a thing? Do people still come on here?",
             "WHAAAAT! That's so insane! Oh my gosh my mind has just been blown :shocked: It's super trippy"],
            # INFJ
            ["Lost, oh so lost",
             "I confess that this is a battle i want to lose."],
            # ESFJ
            ["I'm sorry :(",
             "I'm not sure what's better... "],
        ]
    })

if __name__ == '__main__':
    mock_data = generate_mock_data()  # 生成模拟数据
    result = predict_mbti_simple(mock_data)  # 预测MBTI类型

    # 只取每个 post 列表中的第一句话
    short_posts = [post[0] for post in mock_data['posts']]  # 只取每个帖子中的第一句话

    # 创建表格 DataFrame
    df_output = pd.DataFrame({
        'Post': short_posts,  # 帖子内容
        'Correct Label': ['ESTP', 'ENTP', 'ESFP', 'INFJ', 'ESFJ'],  # 正确标签
        'Predicted Label': result['MBTI']  # 预测标签
    })

    # 打印表格并左对齐
    pd.set_option('display.colheader_justify', 'left')  # 设置表头左对齐
    print(df_output.to_string(index=False, justify='left'))  # 打印输出
