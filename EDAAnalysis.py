from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

def plot_eda(df):
    """执行EDA可视化"""
    plt.figure(figsize=(20, 20))
    # 标签分布
    plt.subplot(3, 2, 1)
    df['E/I'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('E/I Distribution')
    plt.subplot(3, 2, 2)
    sns.histplot(df['avg_post_length'], bins=30, kde=True)
    plt.title('Average Post Length Distribution')
    # 词云生成
    plt.subplot(3, 2, 3)
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(df['combined_text']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Posts')
    # 特征相关性
    plt.subplot(3, 2, 4)
    corr_matrix = df[['post_count', 'avg_post_length',
                     'polarity', 'subjectivity', 'lexical_diversity']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    # 情感分析分布
    plt.subplot(3, 2, 5)
    sns.kdeplot(data=df, x='polarity', hue='source', fill=True)
    plt.title('Polarity Distribution by Source')
    plt.tight_layout()
    plt.show()