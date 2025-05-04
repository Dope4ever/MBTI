# 导入 numpy，用于数值计算和数组操作
import numpy as np

# 导入 NMF（非负矩阵分解）模块，用于从文本中提取主题特征
from sklearn.decomposition import NMF

# 导入 NLTK 库中的 stopwords，用于加载停用词表
# 停用词：指在文本分析中常见的、出现频率非常高但对文本含义贡献很小的词。例如：
# 英语中的停用词：如 "the", "is", "in", "and", "a", "an" 等。
# 中文中的停用词：如 "的", "是", "在", "和" 等。
from nltk.corpus import stopwords

# 导入 NLTK 中的词形还原工具，用于对单词进行词形还原
from nltk.stem import WordNetLemmatizer

# 导入 TF-IDF 向量化器和词袋模型（BoW）向量化器
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 导入 hstack，用于合并稀疏矩阵
# 是将多个稀疏矩阵按列拼接在一起，形成一个新的矩阵
# 稀疏矩阵是指矩阵中的大部分元素为零。只有少数几个元素是非零的。
# 稀疏矩阵只存储非零元素及其位置。
# 稠密矩阵存储所有元素，包括零元素
from scipy.sparse import hstack

# 导入 StandardScaler，用于特征标准化
from sklearn.preprocessing import StandardScaler


def advanced_feature_engineering(df):
    """
    文本预处理和特征工程处理函数
    """

    # 创建词形还原对象，用于将单词还原为其基本形式，如将“running”还原为“run”
    lemma = WordNetLemmatizer()

    # 获取英语的停用词列表，去除常见的无意义的词
    stop_words = set(stopwords.words('english'))

    # 文本预处理函数，去除停用词，进行词形还原，且去掉长度小于等于2的词
    def preprocess(text):
        tokens = [lemma.lemmatize(word) for word in text.split()  # 对每个词进行词形还原
                  if word not in stop_words and len(word) > 2]  # 去除停用词和长度小于等于2的词
        return ' '.join(tokens)  # 将处理后的词拼接回成一个新的字符串

    # 输入：原始文本存储在 'combined_text' 列中。
    # 处理：对每一行文本应用 preprocess 函数，进行文本清洗、去除停用词、词形还原等操作。
    # 输出：将处理后的文本存储在 'processed_text' 列中。
    df['processed_text'] = df['combined_text'].apply(preprocess)

    # TF-IDF 特征提取
    tfidf = TfidfVectorizer(
        max_features=24,  # 选择前 24 个最重要的特征
        # n-grams（n元语法）
        # n=1 是 单个词，即单词的特征。
        # n=2 是 二元词组，即连续的两个词作为一个特征（例如 "machine learning" 这样的词组）。
        ngram_range=(1, 2),  # 使用单个词和二元词组作为特征
        stop_words='english',  # 使用英语的停用词
        #使用 对数缩放 来调整词频，100调整为log100
        sublinear_tf=True  # 启用子线性词频（减少高频词的影响）
    )
    # 将文本转换为 TF-IDF 特征矩阵
    tfidf_features = tfidf.fit_transform(df['processed_text'])

    # 词袋模型特征提取（BoW）
    count_vec = CountVectorizer(
        max_features=10,  # 选择前 10 个最常见的特征
        # 如果一个单词在文本中出现，则它的特征值为 1。
        # 如果一个单词在文本中没有出现，则它的特征值为 0。
        binary=True  # 是否将词频转化为二进制（只考虑词是否出现，而不考虑次数）
    )
    # 将文本转换为词袋模型特征矩阵
    bow_features = count_vec.fit_transform(df['processed_text'])

    # NMF（非负矩阵分解）提取主题特征
    nmf = NMF(
        n_components=3,  # 提取 3 个主题
        max_iter=1000,  # 设置最大迭代次数
        random_state=42,  # 固定随机种子，保证实验可复现
        init='nndsvda'  # 使用 NMF 初始化方法
    )
    # 使用词袋模型特征矩阵进行 NMF 训练，得到每个样本在 3 个主题下的得分
    nmf_features = nmf.fit_transform(bow_features)

    # 提取一些元数据特征
    meta_data = np.array(df[[  # 从 DataFrame 中提取其他数值型特征
        'post_count',  # 帖子数量
        'avg_post_length',  # 平均帖子长度
        'polarity',  # 情感极性
        'subjectivity',  # 主观性
        'lexical_diversity'  # 词汇多样性
    ]]).astype(float)  # 将其转换为浮点类型

    # 对元数据特征进行标准化（均值为 0，方差为 1）
    meta_scaler = StandardScaler()
    meta_features = meta_scaler.fit_transform(meta_data)  # 拟合并转换数据

    # 特征组合，将各个特征矩阵合并成一个大的特征矩阵
    final_features = hstack([  # 使用 hstack 合并多个稀疏矩阵
        tfidf_features,  # TF-IDF 特征
        nmf_features,  # NMF 主题特征
        meta_features  # 标准化后的元数据特征
    ])

    # 返回最终的特征矩阵和特征工程处理对象
    return final_features, {
        'tfidf': tfidf,
        'count_vec': count_vec,
        'nmf': nmf,
        'meta_scaler': meta_scaler  # 返回已拟合的 scaler，用于新数据的转换
    }
