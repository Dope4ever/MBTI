import jsonlines  # 导入处理jsonl格式文件的库
import pandas as pd  # 导入用于数据处理的pandas库
import numpy as np  # 导入用于数值计算的numpy库
import re  # 导入正则表达式库，用于文本清洗
from textblob import TextBlob  # 导入textblob库用于情感分析


def load_data(filepath):
    """加载并预处理原始数据"""
    # 读取jsonl文件，使用jsonlines库来逐行读取数据
    with jsonlines.open(filepath) as reader:
        data = [obj for obj in reader]  # 将每一行数据加载到data列表中
    df = pd.DataFrame(data)  # 将数据转化为DataFrame格式，以便后续操作

    # 合并所有帖子内容，并清理文本
    df['combined_text'] = df['posts'].apply(
        lambda x: ' '.join([clean_text(post) for post in x]))  # 对每个用户的所有帖子进行文本清理后合并

    # 提取MBTI维度标签（E/I, S/N, T/F, J/P）并存储到新的列中
    dimensions = ['E/I', 'S/N', 'T/F', 'J/P']  # 定义MBTI的四个维度
    for dim in dimensions:
        df[dim] = df['hardlabels'].apply(lambda x: x[dim])  # 根据hardlabels字段中的标签提取每个维度的值并存储

    # 添加其他的特征（如情感分析、词汇多样性等）
    df = add_meta_features(df)  # 调用add_meta_features函数添加特征

    # 删除不必要的列，返回清理后的DataFrame
    return df.drop(columns=['posts', 'hardlabels', 'annotation',
                            'softlabels'])  # 删除'posts', 'hardlabels', 'annotation', 'softlabels'列


def clean_text(text):
    """文本清洗函数"""
    text = text.lower()  # 将文本转为小写
    text = re.sub(r'[^\w\s]', '', text)  # 移除文本中的标点符号（只保留字母和数字）
    text = re.sub(r'\d+', '', text)  # 移除文本中的所有数字
    text = re.sub(r'\s+', ' ', text)  # 将多个连续的空格合并为一个空格
    return text.strip()  # 去除文本开头和结尾的空格


def add_meta_features(df):
    """添加元特征"""
    # 文本长度特征：计算每个样本（帖子）的字符数（文本长度）
    df['post_count'] = df['posts'].apply(len)  # 计算每个样本（帖子）中的文本数量
    # 平均帖子长度特征：计算每个样本（帖子）中的平均单词数量
    df['avg_post_length'] = df['posts'].apply(
        lambda x: np.mean([len(p.split()) for p in x]))  # 对每个帖子计算其平均单词数

    # 情感分析特征：通过TextBlob计算文本的情感极性和主观性
    df['polarity'] = df['combined_text'].apply(
        lambda x: TextBlob(x).sentiment.polarity)  # 计算每个帖子的情感极性（positive/negative）
    df['subjectivity'] = df['combined_text'].apply(
        lambda x: TextBlob(x).sentiment.subjectivity)  # 计算每个帖子的主观性（subjective/objective）

    # 词汇多样性特征：计算每个帖子中的词汇多样性
    df['lexical_diversity'] = df['combined_text'].apply(
        lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0)  # 词汇多样性 = 唯一词汇数 / 总词汇数

    return df  # 返回添加了元特征的DataFrame
