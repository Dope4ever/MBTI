# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import torch
import joblib
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    GPT2Config
)
from typing import List
from textblob import TextBlob
from scipy.sparse import hstack


# ----------------------
# 特征处理模块
# ----------------------
class MBTIFeatureProcessor:
    def __init__(self, model_dir: str):
        self.feature_objects = joblib.load(f"{model_dir}/feature_processing_objects.pkl")
        assert hasattr(self.feature_objects['meta_scaler'], 'mean_'), "Scaler未正确训练"
        self.tfidf = self.feature_objects['tfidf']
        self.count_vec = self.feature_objects['count_vec']
        self.nmf = self.feature_objects['nmf']
        self.meta_scaler = self.feature_objects['meta_scaler']

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text.strip()

    def process_text(self, posts: List[str]) -> np.ndarray:
        cleaned_text = ' '.join([self._clean_text(p) for p in posts])
        tfidf_feat = self.tfidf.transform([cleaned_text])
        bow_feat = self.count_vec.transform([cleaned_text])
        nmf_feat = self.nmf.transform(bow_feat)
        meta_features = self._get_meta_features(cleaned_text)
        meta_scaled = self.meta_scaler.transform([meta_features])
        combined = hstack([tfidf_feat, nmf_feat, meta_scaled])
        # 转换为 numpy 数组
        return combined.toarray().squeeze()  # 直接返回 numpy 数组

    def _get_meta_features(self, text: str) -> List[float]:
        try:
            words = text.split()
            word_count = len(words)
            unique_words = len(set(words))
            blob = TextBlob(text)
            return [
                word_count,
                len(text),
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                unique_words / (word_count + 1e-8)
            ]
        except Exception as e:
            print(f"元特征生成失败: {str(e)}")
            return [0.0] * 5


# ----------------------
# 数据集模块
# ----------------------

# ----------------------
# 数据集模块
# ----------------------
class MBTIDataset(Dataset):
    def __init__(self, data_path: str, processor: MBTIFeatureProcessor,
                 tokenizer: GPT2Tokenizer, max_length=256):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length  # 总长度包含条件特征
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
            print(f"成功加载{len(self.data)}条数据")

    # 修改 MBTIDataset 的 collate_fn：
    @staticmethod
    def collate_fn(batch):
        valid_batch = [b for b in batch if b is not None]
        if not valid_batch:
            return {
                'features': torch.zeros((1, 32)),
                'input_ids': torch.zeros((1, 256), dtype=torch.long),
                'attention_mask': torch.zeros((1, 257), dtype=torch.long),  # 确保维度一致
                'labels': torch.zeros((1, 256), dtype=torch.long)
            }
        features = np.stack([b['features'].numpy() for b in valid_batch])
        return {
            'features': torch.from_numpy(features).float(),
            'input_ids': torch.stack([b['input_ids'] for b in valid_batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in valid_batch]),
            'labels': torch.stack([b['labels'] for b in valid_batch])
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            features = self.processor.process_text(item['posts'])
            text = ' '.join(item['posts'])
            # Tokenization处理，预留条件特征位置
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,  # 保证max_length为256
                padding='max_length',  # 已生效 padding_side='left'
                truncation=True,
                return_tensors='pt'
            )
            # 确保特征维度正确
            assert len(features) == 32, f"特征维度应为32，实际得到{len(features)}"

            # 生成对齐的labels（第一个位置用-100忽略）
            input_ids = inputs.input_ids.squeeze()
            labels = torch.cat([
                torch.tensor([-100], dtype=torch.long),  # 忽略条件特征位置
                input_ids
            ])  # 总长度=max_length

            # 生成 attention_mask
            attention_mask = torch.ones_like(input_ids)  # 确保attention_mask与input_ids一致

            return {
                'features': torch.tensor(features, dtype=torch.float32),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        except Exception as e:
            print(f"处理第{idx}条数据失败: {str(e)}")
            return None



# ----------------------
# 改进的模型架构
# ----------------------
class ConditionalGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # 强制设置pad_token_id
        if config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id

        # 条件投影层
        self.condition_proj = torch.nn.Linear(32, config.n_embd)
        self.condition_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.condition_proj.bias.data.zero_()

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        features = kwargs.pop('features', None)
        attention_mask = kwargs.get('attention_mask', None)

        if past_key_values is None:
            # 初始步骤：仅使用特征生成条件嵌入
            cond_emb = self.condition_proj(features).unsqueeze(1)
            inputs_embeds = cond_emb
            attention_mask = torch.ones(
                (cond_emb.size(0), cond_emb.size(1)),
                dtype=torch.long,
                device=cond_emb.device
            )
            return {
                'inputs_embeds': inputs_embeds,
                'attention_mask': attention_mask,
                'input_ids': None,  # 禁用 input_ids
                'features': features  # 传递给后续步骤
            }
        else:
            # 后续步骤：将新生成的 input_ids 转换为嵌入并更新注意力掩码
            if input_ids is not None:
                # 获取当前输入的嵌入
                new_inputs_embeds = self.transformer.wte(input_ids)
                # 合并过去的 past_key_values 对应的嵌入（如果需要）
                # 这里假设 past_key_values 已经处理过历史嵌入，仅扩展当前新 token 的嵌入
                # 注意：此处可能需要根据具体模型结构调整
                inputs_embeds = new_inputs_embeds
                # 更新注意力掩码
                if attention_mask is not None:
                    new_attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((input_ids.size(0), 1), dtype=torch.long, device=input_ids.device)
                    ], dim=1)
                else:
                    new_attention_mask = torch.ones(
                        (input_ids.size(0), input_ids.size(1)),
                        dtype=torch.long,
                        device=input_ids.device
                    )
                return {
                    'inputs_embeds': inputs_embeds,
                    'attention_mask': new_attention_mask,
                    'input_ids': None,  # 禁用 input_ids
                    'past_key_values': past_key_values,
                    'features': features  # 保持特征传递
                }
            else:
                # 如果 input_ids 为空（理论不可能），返回默认值
                return {
                    'inputs_embeds': None,
                    'attention_mask': attention_mask,
                    'input_ids': None,
                    'features': features
                }

    def forward(self, input_ids=None, attention_mask=None, features=None, labels=None, **kwargs):
        if features is not None:
            cond_emb = self.condition_proj(features).unsqueeze(1)
            if input_ids is not None:
                inputs_embeds = torch.cat([cond_emb, self.transformer.wte(input_ids)], dim=1)
            else:
                inputs_embeds = cond_emb
        else:
            inputs_embeds = self.transformer.wte(input_ids) if input_ids is not None else None

        # Remove 'inputs_embeds' from **kwargs to prevent duplication
        if 'inputs_embeds' in kwargs:
            del kwargs['inputs_embeds']

        outputs = super().forward(
            input_ids=None,  # Explicitly set to None to avoid conflict
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        # 手动计算损失（保持原有逻辑）
        if labels is not None:
            shift_logits = outputs.logits[:, :-1, :].contiguous()  # 调整索引逻辑
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            outputs.loss = loss
        return outputs

# ----------------------
# 训练流程
# ----------------------
class MBTITrainer:
    def __init__(self, model_path="./model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, data_path="mbtibench-nolabel.jsonl"):
        # 初始化特征处理器
        feature_processor = MBTIFeatureProcessor(self.model_path)

        # 初始化tokenizer并设置pad_token
        tokenizer = GPT2Tokenizer.from_pretrained(f"{self.model_path}/gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 创建数据集
        dataset = MBTIDataset(
            data_path=data_path,
            processor=feature_processor,
            tokenizer=tokenizer
        )

        # 加载模型配置
        config = GPT2Config.from_pretrained(f"{self.model_path}/gpt2")
        config.n_positions = 1024  # 根据需求调整
        model = ConditionalGPT2.from_pretrained(
            f"{self.model_path}/gpt2",
            config=config,
            ignore_mismatched_sizes=True
        ).to(self.device)

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,  # 降低累积步数
            num_train_epochs=1,
            learning_rate=2e-5,  # 调整学习率
            fp16=torch.cuda.is_available(),
            logging_steps=50,
            save_strategy="steps",
            save_steps=500,
            remove_unused_columns=False,
            label_names=["labels"]  # 显式指定标签名称
        )

        # 初始化训练器
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=dataset.collate_fn,
        )

        # 开始训练
        trainer.train()
        model.save_pretrained(f"{self.model_path}/integrated_gpt2")


# ----------------------
# 生成接口
# ----------------------
class MBTIGenerator:
    def __init__(self, model_path="./model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载组件
        self.tokenizer = GPT2Tokenizer.from_pretrained(f"{model_path}/integrated_gpt2")
        self.model = ConditionalGPT2.from_pretrained(f"{model_path}/integrated_gpt2").to(self.device)
        self.feature_processor = MBTIFeatureProcessor(model_path)
        # 设置生成参数
        self.model.eval()

    def generate(self, posts, max_length=100) -> str:
        features = self.feature_processor.process_text(posts)
        features_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)

        generation_config = {
            "max_length": max_length + 1,  # 补偿条件嵌入的长度
            "temperature": 0.9,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "features": features_tensor  # 直接传递特征
        }

        with torch.no_grad():
            outputs = self.model.generate(**generation_config)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    # 训练模型
    trainer = MBTITrainer()
    trainer.train()