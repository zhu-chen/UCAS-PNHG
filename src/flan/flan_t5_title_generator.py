import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import random
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 使用清华镜像加速Hugging Face下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class PersonalizedTitleGenerator:
    def __init__(self, model_name="google/flan-t5-base", max_length=512):
        """初始化基于FLAN-T5的个性化标题生成器"""
        self.model_name = model_name
        self.max_length = max_length
        
        # UCAS项目目录结构 - FLAN-T5版本
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "..", "scripts", "data", "raw")
        self.output_dir = os.path.join(self.project_root, "..", "..", "results", "flan")
        
        # 加载预训练模型和分词器
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # 设置特殊标记
        self.user_token = "<user>"  # 用户偏好标记
        self.tokenizer.add_tokens([self.user_token])
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 评估指标
        self.rouge_metric = load_metric("rouge")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self):
        """加载UCAS目录下的PENS数据集"""
        logger.info(f"从{self.data_dir}加载新闻语料库和用户行为数据...")
        
        # 定义UCAS目录下的文件路径
        news_path = os.path.join(self.data_dir, "news_corpus.pkl")
        train_path = os.path.join(self.data_dir, "train.pkl")
        val_path = os.path.join(self.data_dir, "validation.pkl")
        test_path = os.path.join(self.data_dir, "test.pkl")
        
        # 加载新闻内容
        self.news_df = pd.read_pickle(news_path)
        
        # 加载用户行为数据
        self.train_df = pd.read_pickle(train_path)
        self.val_df = pd.read_pickle(val_path)
        self.test_df = pd.read_pickle(test_path)
        
        # 构建新闻ID到内容的映射
        self.news_id_to_content = self.news_df.set_index('news_id')[['title', 'body', 'category']].to_dict(orient='index')
        
        logger.info(f"数据加载完成: {len(self.news_df)} 篇新闻，{len(self.train_df)} 条训练样本")
        
        return self
    
    def create_user_profiles(self):
        """创建用户兴趣画像"""
        logger.info("构建用户兴趣画像...")
        
        # 为每个用户收集点击过的新闻类别
        self.user_profiles = defaultdict(lambda: defaultdict(int))
        
        # 从训练集构建用户画像
        for _, row in tqdm(self.train_df.iterrows(), total=len(self.train_df), desc="构建用户画像"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        category = self.news_id_to_content[news_id]['category']
                        self.user_profiles[user_id][category] += 1
        
        # 将计数转换为百分比
        for user_id, profile in self.user_profiles.items():
            total = sum(profile.values())
            if total > 0:
                self.user_profiles[user_id] = {cat: cnt/total for cat, cnt in profile.items()}
        
        logger.info(f"用户画像构建完成，共 {len(self.user_profiles)} 个用户")
        return self
    
    def prepare_datasets(self, sample_size=None):
        """准备训练、验证和测试数据集"""
        logger.info("准备数据集...")
        
        # 构建训练样本
        train_samples = []
        for _, row in tqdm(self.train_df.iterrows(), total=len(self.train_df), desc="准备训练数据"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                # 获取用户画像
                user_profile = self.user_profiles.get(user_id, {})
                
                # 获取用户点击的新闻
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        news = self.news_id_to_content[news_id]
                        # 构建输入文本：用户偏好 + 新闻正文
                        user_pref_text = self._build_user_preference_text(user_profile)
                        input_text = f"生成个性化标题: {user_pref_text} {self.user_token} {news['body']}"
                        
                        # 目标文本：原始标题
                        target_text = news['title']
                        
                        train_samples.append({
                            'user_id': user_id,
                            'news_id': news_id,
                            'input_text': input_text,
                            'target_text': target_text
                        })
        
        # 构建验证样本
        val_samples = []
        for _, row in tqdm(self.val_df.iterrows(), total=len(self.val_df), desc="准备验证数据"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                user_profile = self.user_profiles.get(user_id, {})
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        news = self.news_id_to_content[news_id]
                        user_pref_text = self._build_user_preference_text(user_profile)
                        input_text = f"生成个性化标题: {user_pref_text} {self.user_token} {news['body']}"
                        target_text = news['title']
                        
                        val_samples.append({
                            'user_id': user_id,
                            'news_id': news_id,
                            'input_text': input_text,
                            'target_text': target_text
                        })
        
        # 构建测试样本
        test_samples = []
        for _, row in tqdm(self.test_df.iterrows(), total=len(self.test_df), desc="准备测试数据"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                user_profile = self.user_profiles.get(user_id, {})
                clicked_news_ids = str(row.get('clicknewsID', '')).split(',')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        news = self.news_id_to_content[news_id]
                        user_pref_text = self._build_user_preference_text(user_profile)
                        input_text = f"生成个性化标题: {user_pref_text} {self.user_token} {news['body']}"
                        target_text = news['title']
                        
                        test_samples.append({
                            'user_id': user_id,
                            'news_id': news_id,
                            'input_text': input_text,
                            'target_text': target_text
                        })
        
        # 随机采样（如果需要）
        if sample_size and sample_size < len(train_samples):
            train_samples = random.sample(train_samples, sample_size)
        if sample_size and sample_size < len(val_samples):
            val_samples = random.sample(val_samples, min(sample_size, len(val_samples)))
        if sample_size and sample_size < len(test_samples):
            test_samples = random.sample(test_samples, min(sample_size, len(test_samples)))
        
        # 转换为Dataset对象
        self.train_dataset = Dataset.from_list(train_samples)
        self.val_dataset = Dataset.from_list(val_samples)
        self.test_dataset = Dataset.from_list(test_samples)
        
        logger.info(f"数据集准备完成: 训练集={len(self.train_dataset)}, 验证集={len(self.val_dataset)}, 测试集={len(self.test_dataset)}")
        return self
    
    def _build_user_preference_text(self, user_profile):
        """构建用户偏好文本表示"""
        if not user_profile:
            return "用户兴趣: 无明确偏好"
        
        # 按兴趣程度排序
        sorted_preferences = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)
        
        # 取前3个兴趣类别
        top_preferences = sorted_preferences[:3]
        
        # 构建文本表示
        pref_text = "用户兴趣: "
        for category, score in top_preferences:
            pref_text += f"{category}({int(score*100)}%), "
        
        return pref_text.rstrip(', ')
    
    def preprocess_data(self):
        """预处理数据集"""
        logger.info("预处理数据集...")
        
        def tokenize_function(examples):
            # 编码输入文本
            inputs = self.tokenizer(
                examples["input_text"], 
                max_length=self.max_length, 
                truncation=True,
                padding="max_length"
            )
            
            # 编码目标文本
            targets = self.tokenizer(
                examples["target_text"], 
                max_length=64,  # 标题通常较短
                truncation=True
            )
            
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        # 对数据集进行预处理
        self.train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        self.val_dataset = self.val_dataset.map(tokenize_function, batched=True)
        self.test_dataset = self.test_dataset.map(tokenize_function, batched=True)
        
        # 设置格式以便与PyTorch兼容
        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        self.test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        logger.info("数据预处理完成")
        return self
    
    def train(self, batch_size=4, num_epochs=3, learning_rate=5e-5):
        """微调FLAN-T5模型，结果保存到UCAS指定目录"""
        logger.info(f"开始微调FLAN-T5模型，batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
        
        # 定义训练参数，输出到UCAS结果目录
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            fp16=torch.cuda.is_available()  # 使用混合精度训练加速
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        # 定义评估指标
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # 解码预测结果
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # 解码目标文本
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # 计算ROUGE分数
            result = self.rouge_metric.compute(
                predictions=decoded_preds, 
                references=decoded_labels, 
                use_stemmer=True
            )
            
            # 提取ROUGE-1, ROUGE-2, ROUGE-L的f1分数
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            
            # 添加生成的平均长度
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)
            
            return {k: round(v, 4) for k, v in result.items()}
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 开始训练
        trainer.train()
        
        # 保存最佳模型到UCAS结果目录
        best_model_dir = os.path.join(self.output_dir, "best_model")
        trainer.save_model(best_model_dir)
        self.tokenizer.save_pretrained(best_model_dir)
        
        logger.info(f"模型训练完成，已保存至 {best_model_dir}")
        return self
    
    def generate_personalized_title(self, user_id, news_id, max_length=64):
        """为特定用户生成个性化新闻标题"""
        # 获取用户画像
        user_profile = self.user_profiles.get(user_id, {})
        
        # 获取新闻内容
        if news_id not in self.news_id_to_content:
            logger.error(f"新闻ID {news_id} 不存在于 {self.data_dir}")
            return None
        
        news = self.news_id_to_content[news_id]
        
        # 构建输入文本
        user_pref_text = self._build_user_preference_text(user_profile)
        input_text = f"生成个性化标题: {user_pref_text} {self.user_token} {news['body']}"
        
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=self.max_length, truncation=True)
        
        # 生成标题
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"].to(self.model.device),
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # 解码生成的标题
        generated_title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            'user_id': user_id,
            'news_id': news_id,
            'original_title': news['title'],
            'generated_title': generated_title,
            'user_preferences': user_profile
        }
    
    def evaluate(self, sample_size=100):
        """评估模型性能，结果保存到UCAS目录"""
        logger.info("开始模型评估...")
        
        # 随机选择样本进行评估
        if sample_size < len(self.test_dataset):
            indices = random.sample(range(len(self.test_dataset)), sample_size)
            samples = [self.test_dataset[i] for i in indices]
        else:
            samples = self.test_dataset
        
        results = []
        for sample in tqdm(samples, desc="生成评估结果"):
            user_id = sample['user_id']
            news_id = sample['news_id']
            
            # 生成个性化标题
            result = self.generate_personalized_title(user_id, news_id)
            if result:
                results.append(result)
        
        # 计算评估指标
        references = [result['original_title'] for result in results]
        predictions = [result['generated_title'] for result in results]
        
        # 计算ROUGE分数
        rouge_scores = self.rouge_metric.compute(
            predictions=predictions, 
            references=references, 
            use_stemmer=True
        )
        
        # 提取ROUGE-1, ROUGE-2, ROUGE-L的f1分数
        rouge_scores = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}
        
        logger.info(f"评估完成: ROUGE-1={rouge_scores['rouge1']:.2f}, ROUGE-2={rouge_scores['rouge2']:.2f}, ROUGE-L={rouge_scores['rougeL']:.2f}")
        
        # 保存示例结果到UCAS目录
        sample_results_path = os.path.join(self.output_dir, "sample_results.txt")
        with open(sample_results_path, "w", encoding="utf-8") as f:
            f.write("个性化新闻标题生成示例:\n\n")
            for i, result in enumerate(results[:10]):
                f.write(f"示例 {i+1}:\n")
                f.write(f"用户ID: {result['user_id']}\n")
                f.write(f"用户偏好: {result['user_preferences']}\n")
                f.write(f"原始标题: {result['original_title']}\n")
                f.write(f"生成标题: {result['generated_title']}\n")
                f.write("-" * 50 + "\n\n")
        
        return {
            'rouge_scores': rouge_scores,
            'sample_results': results[:10]
        }

# 主函数
if __name__ == "__main__":
    # 初始化生成器
    # 可以选择使用不同的FLAN-T5模型：google/flan-t5-small, google/flan-t5-base, google/flan-t5-large
    generator = PersonalizedTitleGenerator(model_name="google/flan-t5-base")
    
    # 加载UCAS目录下的数据
    generator.load_data()
    
    # 创建用户画像
    generator.create_user_profiles()
    
    # 准备数据集（使用10000个样本进行训练，加速测试）
    generator.prepare_datasets(sample_size=10000)
    
    # 预处理数据
    generator.preprocess_data()
    
    # 训练模型
    generator.train(batch_size=4, num_epochs=3)
    
    # 评估模型
    evaluation_results = generator.evaluate()
    
    # 示例：为特定用户生成个性化标题
    if generator.test_dataset:
        sample_user_id = generator.test_dataset[0]['user_id']
        sample_news_id = generator.test_dataset[0]['news_id']
        
        personalized_title = generator.generate_personalized_title(sample_user_id, sample_news_id)
        print("\n个性化标题生成示例:")
        print(f"用户ID: {personalized_title['user_id']}")
        print(f"新闻ID: {personalized_title['news_id']}")
        print(f"原始标题: {personalized_title['original_title']}")
        print(f"生成标题: {personalized_title['generated_title']}")
        print(f"用户偏好: {personalized_title['user_preferences']}")
    else:
        logger.warning("测试数据集为空，无法展示生成示例")