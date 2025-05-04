import torch  # 导入PyTorch库，用于深度学习模型的训练和推理
from transformers import GPT2Tokenizer  # 从transformers库中导入GPT2Tokenizer类，用于处理GPT-2模型的文本
import numpy as np  # 导入NumPy库，用于数值计算和处理数组

# 引入自定义模块（假设在项目中有对应的文件），其中包含了ConditionalGPT2和MBTIFeatureProcessor类
from GPT2ModelPredict import ConditionalGPT2, MBTIFeatureProcessor


class MBTIGenerator:
    def __init__(self, model_path="./model"):
        """
        初始化MBTIGenerator类的实例
        :param model_path: 预训练模型的路径，默认是"./model"
        """
        self.model_path = model_path  # 设置模型路径
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有GPU，若有则使用GPU，否则使用CPU
        self.tokenizer = GPT2Tokenizer.from_pretrained(f"{model_path}/integrated_gpt2")  # 从指定路径加载GPT-2的tokenizer
        self.tokenizer.padding_side = 'left'  # 设置填充的方向为左边
        if self.tokenizer.pad_token is None:  # 如果tokenizer没有定义pad_token，则使用eos_token作为pad_token
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # 设置pad_token的ID为eos_token的ID
        self.tokenizer.eos_token_id = self.tokenizer.eos_token_id  # 设置eos_token的ID

        # 加载自定义的ConditionalGPT2模型，并将其转移到对应设备（GPU/CPU）
        self.model = ConditionalGPT2.from_pretrained(f"{model_path}/integrated_gpt2").to(self.device)

        # 初始化MBTIFeatureProcessor，用于处理MBTI特征
        self.feature_processor = MBTIFeatureProcessor(model_path)
        self.model.eval()  # 将模型设置为评估模式，关闭dropout等训练相关的操作

    def generate(self, posts, max_length=1024) -> str:
        """
        生成基于MBTI特征的文本
        :param posts: 输入的文本列表（可能是MBTI类型的语句）
        :param max_length: 生成文本的最大长度，避免超过模型的最大限制
        :return: 生成的文本字符串
        """
        features = self.feature_processor.process_text(posts)  # 使用MBTIFeatureProcessor处理输入的文本
        # 将处理后的特征转换为PyTorch张量，并转移到对应的设备（GPU/CPU）
        features_tensor = torch.tensor(np.array([features]), dtype=torch.float32).to(self.device)

        # 配置生成文本的参数
        generation_config = {
            "max_length": max_length,  # 控制最大长度，避免超过模型的最大限制
            "temperature": 0.5,  # 降低温度，生成更确定性的文本
            "top_p": 0.9,  # 保持一定的多样性，但不至于过于随机
            "repetition_penalty": 1.2,  # 增加重复词汇的惩罚，避免生成重复的词
            "num_return_sequences": 1,  # 只返回一个生成文本
            "pad_token_id": self.tokenizer.pad_token_id,  # 使用tokenizer定义的pad_token_id
            "eos_token_id": self.tokenizer.eos_token_id,  # 使用tokenizer定义的eos_token_id
            "do_sample": False,  # 禁用采样，生成更有规律的文本
            "features": features_tensor,  # 将MBTI特征传递给生成器
            "attention_mask": torch.ones((features_tensor.size(0), features_tensor.size(1)), device=self.device)
            # 注意力掩码，确保模型注意到所有输入特征
        }

        with torch.no_grad():  # 禁用梯度计算，加速推理过程
            outputs = self.model.generate(**generation_config)  # 使用模型生成文本
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)  # 解码生成的文本，并跳过特殊的token（如[EOS]等）


if __name__ == "__main__":  # 如果脚本是作为主程序运行
    generator = MBTIGenerator()  # 创建MBTIGenerator实例
    sample_posts = [
        # 输入的样例文本（此处为ESTP类型的示例文本）
        "I'd also be willing, if needed.",
    ]
    print("生成结果:", generator.generate(sample_posts))  # 调用生成方法，并输出生成的结果
