# -*- coding: utf-8 -*-  # 指定文件的编码格式为 UTF-8，确保处理中文字符时不出现乱码

# 导入所需的模块和库
import json  # 用于加载和解析 JSON 格式的数据
import re  # 提供正则表达式操作，用于文本清洗
import numpy as np  # 用于进行数组和矩阵操作，常用于数据科学和机器学习中
import torch  # 用于深度学习，提供张量操作和模型训练等功能
import joblib  # 用于序列化和反序列化 Python 对象，通常用于保存训练好的模型
from torch.utils.data import Dataset  # 用于自定义数据集类，PyTorch 提供的工具
from transformers import (
    GPT2LMHeadModel,  # GPT-2 语言模型头（生成任务）
    GPT2Tokenizer,     # GPT-2 的 tokenizer，用于将文本转换为模型可理解的格式
    Trainer,           # 用于简化训练过程的工具类
    TrainingArguments, # 配置训练参数（如学习率、batch size 等）
    GPT2Config         # GPT-2 模型的配置类，用于定义模型的结构
)
from typing import List  # 用于类型注解，标明某些函数参数或返回值是 List 类型
from textblob import TextBlob  # 用于情感分析，提供对文本情感的分析工具
from scipy.sparse import hstack  # 用于合并稀疏矩阵（例如合并多个特征矩阵）


# ----------------------
# 特征处理模块
# ----------------------
class MBTIFeatureProcessor:
    # 初始化函数，加载训练过程中保存的特征处理对象
    def __init__(self, model_dir: str):
        # 从指定目录加载特征处理对象（包含训练好的 TF-IDF、CountVectorizer、NMF、Scaler 等）
        self.feature_objects = joblib.load(f"{model_dir}/feature_processing_objects.pkl")

        # 确保加载的 'meta_scaler' 对象已经正确训练过（它应该有 'mean_' 属性）
        assert hasattr(self.feature_objects['meta_scaler'], 'mean_'), "Scaler未正确训练"

        # 提取并保存模型的不同特征处理对象
        self.tfidf = self.feature_objects['tfidf']  # TF-IDF 变换器
        self.count_vec = self.feature_objects['count_vec']  # 词频变换器
        self.nmf = self.feature_objects['nmf']  # 非负矩阵分解器（NMF）
        self.meta_scaler = self.feature_objects['meta_scaler']  # 元特征标准化器

    # 静态方法，用于清洗文本数据
    @staticmethod
    def _clean_text(text: str) -> str:
        # 将文本转换为小写
        text = text.lower()
        # 移除文本中的标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 移除文本中的数字
        text = re.sub(r'\d+', '', text)
        # 返回清洗后的文本
        return text.strip()

    # 处理文本数据，提取特征
    def process_text(self, posts: List[str]) -> np.ndarray:
        # 合并所有帖子，并对其进行清洗
        cleaned_text = ' '.join([self._clean_text(p) for p in posts])

        # 提取 TF-IDF 特征
        tfidf_feat = self.tfidf.transform([cleaned_text])
        # 提取词袋模型特征（Bag of Words）
        bow_feat = self.count_vec.transform([cleaned_text])
        # 提取 NMF 特征
        nmf_feat = self.nmf.transform(bow_feat)
        # 提取元特征（例如，文本长度、情感等）
        meta_features = self._get_meta_features(cleaned_text)
        # 对元特征进行标准化
        meta_scaled = self.meta_scaler.transform([meta_features])

        # 合并所有特征（TF-IDF，NMF，元特征）
        combined = hstack([tfidf_feat, nmf_feat, meta_scaled])

        # 将合并后的特征转换为 numpy 数组并返回（去除冗余的维度）
        return combined.toarray().astype(np.float32).squeeze()

    # 提取文本的元特征，如单词计数、情感分析等
    def _get_meta_features(self, text: str) -> List[float]:
        try:
            # 分割文本为单词列表
            words = text.split()
            word_count = len(words)  # 计算单词数
            unique_words = len(set(words))  # 计算独立单词数
            blob = TextBlob(text)  # 使用 TextBlob 进行情感分析
            # 返回元特征列表
            return [
                word_count,  # 单词数
                len(text),  # 文本长度
                blob.sentiment.polarity,  # 情感极性
                blob.sentiment.subjectivity,  # 情感主观性
                unique_words / (word_count + 1e-8)  # 词汇多样性（独立单词/总单词数）
            ]
        except Exception as e:
            print(f"元特征生成失败: {str(e)}")  # 如果提取元特征时出错，则打印错误信息
            # 返回默认的元特征（全为 0.0）
            return [0.0] * 5


# ----------------------
# 数据集模块
# ----------------------
class MBTIDataset(Dataset):
    # 初始化函数，负责加载数据和初始化处理器
    def __init__(self, data_path: str, processor: MBTIFeatureProcessor,
                 tokenizer: GPT2Tokenizer, max_length=256):
        self.processor = processor  # 保存特征处理器（MBTIFeatureProcessor实例）
        self.tokenizer = tokenizer  # 保存GPT2的tokenizer
        self.max_length = max_length  # 设置最大长度（包含条件特征）

        # 读取数据集，数据为JSONL格式，每行一个JSON对象
        with open(data_path, 'r', encoding='utf-8') as f:
            # 解析每行JSON并将其添加到data列表中
            self.data = [json.loads(line) for line in f]
            # 打印加载数据的条数
            print(f"成功加载{len(self.data)}条数据")

    # 静态方法，定义如何处理批量数据
    @staticmethod
    def collate_fn(batch):
        # 过滤出非空的batch元素
        valid_batch = [b for b in batch if b is not None]

        # 如果没有有效的batch，返回默认的全零数据
        if not valid_batch:
            return {
                'features': torch.zeros((1, 32)),
                'input_ids': torch.zeros((1, 256), dtype=torch.long),
                'attention_mask': torch.zeros((1, 257), dtype=torch.long),
                'labels': torch.zeros((1, 256), dtype=torch.long)
            }

        # 将有效的batch数据整合并返回
        return {
            'features': torch.stack([b['features'] for b in valid_batch]),  # 特征数据
            'input_ids': torch.stack([b['input_ids'] for b in valid_batch]),  # 输入ID
            'attention_mask': torch.stack([b['attention_mask'] for b in valid_batch]),  # 注意力掩码
            'labels': torch.stack([b['labels'] for b in valid_batch])  # 标签
        }

    # 返回数据集的长度（即数据的条数）
    def __len__(self):
        return len(self.data)

    # 根据索引获取数据集中的一个样本
    def __getitem__(self, idx):
        try:
            # 获取索引对应的样本数据
            item = self.data[idx]
            # 使用特征处理器处理文本数据并获得特征
            features = self.processor.process_text(item['posts'])
            # 合并所有的帖子文本
            text = ' '.join(item['posts'])

            # 使用tokenizer将文本转换为输入ID，设置最大长度（减去1以为条件特征腾出空间）
            inputs = self.tokenizer(
                text,
                max_length=self.max_length - 1,  # 调整为总长度减去条件特征位置
                padding='max_length',  # 填充到最大长度
                truncation=True,  # 截断超出长度的部分
                return_tensors='pt'  # 返回PyTorch tensor格式
            )

            # 确保提取的特征维度正确
            assert len(features) == 32, f"特征维度应为32，实际得到{len(features)}"

            # 生成对齐的labels，忽略条件特征位置（-100表示忽略）
            input_ids = inputs.input_ids.squeeze()  # 获取输入ID
            labels = torch.cat([torch.tensor([-100], dtype=torch.long),  # 第一个位置的标签用-100忽略
                                input_ids])  # 将输入ID与标签拼接成完整的标签

            # 返回一个字典，包含特征、输入ID、注意力掩码和标签
            return {
                'features': torch.tensor(features, dtype=torch.float32),  # 特征
                'input_ids': input_ids,  # 输入ID
                'attention_mask': inputs.attention_mask.squeeze(),  # 注意力掩码
                'labels': labels  # 标签
            }
        except Exception as e:
            print(f"处理第{idx}条数据失败: {str(e)}")  # 如果出现异常，打印错误信息
            return None  # 返回None，表示该样本无法处理


# ----------------------
# 改进的模型架构
# ----------------------
class ConditionalGPT2(GPT2LMHeadModel):
    # 构造函数，初始化模型架构
    def __init__(self, config):
        super().__init__(config)  # 调用GPT2LMHeadModel的构造函数
        # 强制设置pad_token_id
        if config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id  # 如果没有定义pad_token_id，则设置为eos_token_id

        # 添加条件投影层，用于将输入的特征映射到与模型嵌入空间相同的维度
        # 这部分代码添加了一个条件投影层，用于将输入的32维特征向量（例如：人格特征向量）
        # 映射到与GPT-2模型嵌入空间相同的维度（config.n_embd）。
        # 这使得GPT-2模型能够处理这些特征并将其融入到文本生成中。
        #
        self.condition_proj = torch.nn.Linear(32, config.n_embd)
        #
        #
        #
        self.condition_proj.weight.data.normal_(mean=0.0, std=0.02)  # 初始化权重
        self.condition_proj.bias.data.zero_()  # 初始化偏置为零

    # 用于生成输入数据时处理特征和过去的状态（如历史注意力状态）
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        features = kwargs.pop('features', None)  # 从kwargs中提取features
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, **kwargs)  # 调用父类的生成方法
        if features is not None and past_key_values is None:
            # 如果有特征且没有past_key_values，则生成初始的嵌入和注意力掩码
            # 通过投影层生成条件嵌入

            cond_emb = self.condition_proj(features).unsqueeze(1)
            inputs_embeds = cond_emb  # 使用条件嵌入作为输入嵌入
            attention_mask = torch.ones(
                (cond_emb.size(0), cond_emb.size(1)),
                dtype=torch.long,
                device=cond_emb.device
            )  # 创建注意力掩码，大小与条件嵌入一致
            # 更新输入数据
            inputs.update({
                'inputs_embeds': inputs_embeds,  # 使用条件嵌入作为输入
                'attention_mask': attention_mask,  # 更新注意力掩码
                'input_ids': None  # 禁用input_ids，使用inputs_embeds
            })
        else:
            # 后续步骤使用常规处理
            inputs_embeds = inputs.get('inputs_embeds')  # 获取输入嵌入
            if inputs_embeds is not None:
                # 如果存在inputs_embeds，则调整注意力掩码
                inputs['attention_mask'] = torch.cat([torch.ones((inputs_embeds.size(0), 1), device=inputs_embeds.device),
                                                       inputs.get('attention_mask', torch.ones_like(inputs_embeds[:, 1:]))], dim=1)

        return inputs  # 返回更新后的输入数据

    # 前向传播方法，处理训练和生成的输入
    def forward(self, input_ids=None, attention_mask=None, features=None, labels=None, **kwargs):
        # 如果有特征数据
        if features is not None:
            cond_emb = self.condition_proj(features).unsqueeze(1)  # 通过条件投影层生成条件嵌入
            if input_ids is not None:
                # 训练时：将条件嵌入和输入嵌入拼接
                # 这部分代码在训练时，将生成的人格特征嵌入（cond_emb）与
                # GPT-2的输入嵌入（self.transformer.wte(input_ids)）
                # 拼接在一起，从而将这些特征与文本输入共同提供给GPT-2。
                inputs_embeds = torch.cat([cond_emb, self.transformer.wte(input_ids)], dim=1)
                # 调整attention_mask
                if attention_mask is not None:
                    attention_mask = torch.cat([  # 拼接注意力掩码
                        torch.ones(input_ids.size(0), 1, dtype=torch.long, device=cond_emb.device),
                        attention_mask
                    ], dim=1)
            else:
                # 生成时：仅使用条件嵌入
                #
                #
                inputs_embeds = cond_emb
                attention_mask = torch.ones(
                    (cond_emb.size(0), cond_emb.size(1)),
                    dtype=torch.long,
                    device=cond_emb.device
                )
        else:
            # 如果没有特征数据，使用常规输入
            inputs_embeds = None
            attention_mask = attention_mask

        # 调用父类的forward方法，进行常规的GPT2模型前向传播
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 手动计算损失（与GPT2的标准前向传播逻辑一致）
        if labels is not None:
            shift_logits = outputs.logits[..., 1:, :].contiguous()  # 将logits向前移位，以便计算损失
            shift_labels = labels[..., 1:].contiguous()  # 同样地移位标签
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)  # 使用交叉熵损失
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs.loss = loss  # 设置损失

        return outputs  # 返回模型输出，包括logits和loss


# ----------------------
# 训练流程
# ----------------------
class MBTITrainer:
    def __init__(self, model_path="./model"):
        # 初始化 MBTITrainer 类，设置模型路径
        self.model_path = model_path
        # 设置训练设备：如果有 GPU 则使用 GPU，否则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, data_path="mbtibench-nolabel.jsonl"):
        # 初始化特征处理器，加载预处理对象（特征提取和标准化）
        feature_processor = MBTIFeatureProcessor(self.model_path)

        # 初始化 GPT2 的 Tokenizer 并设置 pad_token
        # 填充（padding）操作，它确保了所有句子都具有相同的长度，
        tokenizer = GPT2Tokenizer.from_pretrained(f"{self.model_path}/gpt2")
        if tokenizer.pad_token is None:
            # 如果没有 pad_token，使用 eos_token 作为 pad_token
            # eos_token 是 GPT-2 模型中的特殊标记，表示序列的结束
            tokenizer.pad_token = tokenizer.eos_token


        # 创建数据集对象，加载数据并处理
        dataset = MBTIDataset(
            data_path=data_path,   # 数据文件路径
            processor=feature_processor,  # 使用的特征处理器
            tokenizer=tokenizer     # tokenizer 会将其转换为一个数字序列
        )

        # 加载模型配置，设置模型的 pad_token_id（确保与 tokenizer 配置一致）
        config = GPT2Config.from_pretrained(f"{self.model_path}/gpt2")
        # tokenizer.pad_token_id 是 0（表示填充用 0），
        # 而 config.pad_token_id 是 1（表示模型使用 1 来表示填充）
        config.pad_token_id = tokenizer.pad_token_id

        # 初始化模型，使用预训练的 GPT2 模型，并结合上面加载的配置
        model = ConditionalGPT2.from_pretrained(
            f"{self.model_path}/gpt2",  # 模型的路径
            config=config,              # 配置文件
            ignore_mismatched_sizes=True  # 允许忽略尺寸不匹配的权重
        ).to(self.device)  # 将模型加载到指定的设备（GPU/CPU）

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir="./results",  # 训练结果保存路径
            per_device_train_batch_size=2,#每个设备上的批次大小
            # 先“积累”多个小批次的梯度，直到完成一个大的批次，最后再更新模型。
            gradient_accumulation_steps=8,  # 设置梯度累积步数，避免显存溢出
            num_train_epochs=1,  # 设置训练轮数
            learning_rate=2e-5, # 设置学习率为 2e-5
            fp16=torch.cuda.is_available(),  # 如果有 GPU 且支持半精度运算，则启用 fp16
            logging_steps=50,  # 每50步打印一次日志
            logging_dir="./logs",  # 日志保存路径
            save_strategy="steps",  # 设置按步数保存模型
            save_steps=500,  # 每500步保存一次模型
            remove_unused_columns=False,  # 不移除数据集中未使用的列
            label_names=["labels"],  # 显式指定标签名称，保证一致性
        )

        # 初始化训练器，传入模型、训练参数、训练数据集、数据处理器等
        trainer = Trainer(
            model=model,  # 传入训练的模型
            args=training_args,  # 训练参数
            train_dataset=dataset,  # 训练数据集
            data_collator=dataset.collate_fn,  # 数据集的 collate_fn，用于批处理时合并数据
        )

        # 开始训练并打印损失信息
        trainer.train()  # 启动训练
        # 训练完成后，保存训练好的模型
        model.save_pretrained(f"{self.model_path}/integrated_gpt2")


# ----------------------
# 生成接口
# ----------------------
class MBTIGenerator:
    def __init__(self, model_path="./model"):
        # 初始化生成器时指定模型路径，默认路径为"./model"
        self.model_path = model_path
        # 设置设备：如果有 GPU 则使用 GPU，否则使用 CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 tokenizer 和训练好的模型
        self.tokenizer = GPT2Tokenizer.from_pretrained(f"{model_path}/integrated_gpt2")
        self.model = ConditionalGPT2.from_pretrained(f"{model_path}/integrated_gpt2").to(self.device)
        # 加载特征处理器，用于生成特征
        self.feature_processor = MBTIFeatureProcessor(model_path)

        # 设置生成模式，模型在生成时不进行梯度更新
        self.model.eval()

    def generate(self, posts: List[str], max_length=100) -> str:
        """生成符合人格特征的文本"""
        # 使用特征处理器生成与输入帖子对应的人格特征向量
        features = self.feature_processor.process_text(posts)
        # 将特征向量转换为 Tensor 并送到指定设备（GPU 或 CPU）
        features_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)

        # 配置生成文本的参数
        generation_config = {
            "max_length": max_length,  # 设置生成文本的最大长度
            # GPT-2 默认的 temperature 值是 1.0，表示生成完全随机化的文本。
            "temperature": 0.9,  # 控制生成文本的多样性（值越高，生成文本越随机）
            "top_p": 0.95,  # 控制 nucleus sampling（控制生成文本的概率分布）
            # 为什么这样设置：repetition_penalty=1.2 表示在生成过程中，对重复词汇施加一定的惩罚，
            # 使得模型倾向于避免重复生成相同的词或短语，从而使生成文本更自然、更富有变化。
            "repetition_penalty": 1.2,  # 防止生成重复的内容
            "num_return_sequences": 1,  # 生成的文本数量
            "pad_token_id": self.tokenizer.pad_token_id  # 显式设置 pad_token_id
        }

        # 使用 torch.no_grad() 以关闭梯度计算（节省内存和计算资源）
        with torch.no_grad():
            # 调用模型的 generate 方法生成文本，传入条件特征（features_tensor）
            outputs = self.model.generate(
                inputs=None,  # 在此生成时，没有传统的 input_ids 输入
                features=features_tensor,  # 使用特征作为条件输入
                **generation_config  # 解包并传入生成配置
            )

        # 解码生成的文本，并去除特殊符号
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    # 训练模型
    trainer = MBTITrainer()
    trainer.train()
