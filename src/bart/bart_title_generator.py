import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, load_metric
from transformers import (
    BartForConditionalGeneration, 
    BartTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
import logging
import gc
import tqdm
import random
from collections import defaultdict

# 设置日志级别为INFO，显示完整中文日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 使用清华镜像加速Hugging Face下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class PersonalizedTitleGenerator:
    def __init__(self, model_name="facebook/bart-base", max_length=384):
        """初始化个性化标题生成器，优化内存使用"""
        self.model_name = model_name
        self.max_length = max_length
        
        # UCAS项目目录结构
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "..", "scripts", "data", "raw")
        self.output_dir = os.path.join(self.project_root, "..", "..", "results", "bart")
        
        # 加载预训练模型和分词器
        self.tokenizer = BartTokenizer.from_pretrained(
            model_name,
            timeout=120  # 增加超时时间，避免下载中断
        )
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # 设置特殊标记
        self.user_token = "<user>"  # 用户偏好标记
        self.tokenizer.add_tokens([self.user_token])
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 评估指标
        self.rouge_metric = load_metric("rouge")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"初始化完成，模型保存路径：{self.output_dir}")
    
    def load_data(self):
        """加载UCAS目录下的PENS数据集"""
        logger.info(f"开始从{self.data_dir}加载新闻语料库和用户行为数据...")
        
        # 定义UCAS目录下的文件路径
        news_path = os.path.join(self.data_dir, "news_corpus.pkl")
        train_path = os.path.join(self.data_dir, "train.pkl")
        val_path = os.path.join(self.data_dir, "validation.pkl")
        test_path = os.path.join(self.data_dir, "test.pkl")
        
        # 加载新闻内容 - 使用低内存模式
        self.news_df = pd.read_pickle(news_path, compression='infer')
        
        # 加载用户行为数据 - 使用低内存模式
        self.train_df = pd.read_pickle(train_path, compression='infer')
        self.val_df = pd.read_pickle(val_path, compression='infer')
        self.test_df = pd.read_pickle(test_path, compression='infer')
        
        # 构建新闻ID到内容的映射
        self.news_id_to_content = self.news_df.set_index('news_id')[['title', 'body', 'category']].to_dict(orient='index')
        
        logger.info(f"数据加载完成: {len(self.news_df)} 篇新闻，{len(self.train_df)} 条训练样本")
        
        return self
    
    def create_user_profiles(self):
        """创建用户兴趣画像，优化内存使用"""
        logger.info("开始构建用户兴趣画像...")
        
        # 为每个用户收集点击过的新闻类别
        self.user_profiles = defaultdict(lambda: defaultdict(int))
        
        # 从训练集构建用户画像 - 使用迭代器减少内存
        for _, row in tqdm.tqdm(self.train_df.iterrows(), total=len(self.train_df), desc="构建用户画像"):
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
        
        # 释放原始DataFrame内存
        del self.train_df
        gc.collect()
        logger.info("已释放训练集DataFrame内存")
        
        return self
    
    def prepare_datasets(self, train_ratio=0.01, val_ratio=0.01, test_ratio=0.01):
        """准备训练、验证和测试数据集，使用比例进行采样"""
        logger.info(f"开始准备数据集，采样比例: 训练集={train_ratio*100:.1f}%, 验证集={val_ratio*100:.1f}%, 测试集={test_ratio*100:.1f}%")
        
        # 构建训练样本 - 使用生成器表达式减少内存
        train_samples = []
        for _, row in tqdm.tqdm(self.val_df.iterrows(), total=len(self.val_df), desc="准备训练数据"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                user_profile = self.user_profiles.get(user_id, {})
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        news = self.news_id_to_content[news_id]
                        user_pref_text = self._build_user_preference_text(user_profile)
                        input_text = f"{user_pref_text} {self.user_token} {news['body']}"
                        target_text = news['title']
                        train_samples.append({
                            'user_id': user_id,
                            'news_id': news_id,
                            'input_text': input_text,
                            'target_text': target_text
                        })
        
        # 构建验证样本
        val_samples = []
        for _, row in tqdm.tqdm(self.val_df.iterrows(), total=len(self.val_df), desc="准备验证数据"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                user_profile = self.user_profiles.get(user_id, {})
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        news = self.news_id_to_content[news_id]
                        user_pref_text = self._build_user_preference_text(user_profile)
                        input_text = f"{user_pref_text} {self.user_token} {news['body']}"
                        target_text = news['title']
                        val_samples.append({
                            'user_id': user_id,
                            'news_id': news_id,
                            'input_text': input_text,
                            'target_text': target_text
                        })
        
        # 构建测试样本
        test_samples = []
        for _, row in tqdm.tqdm(self.test_df.iterrows(), total=len(self.test_df), desc="准备测试数据"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                user_profile = self.user_profiles.get(user_id, {})
                clicked_news_ids = str(row.get('clicknewsID', '')).split(',')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        news = self.news_id_to_content[news_id]
                        user_pref_text = self._build_user_preference_text(user_profile)
                        input_text = f"{user_pref_text} {self.user_token} {news['body']}"
                        target_text = news['title']
                        test_samples.append({
                            'user_id': user_id,
                            'news_id': news_id,
                            'input_text': input_text,
                            'target_text': target_text
                        })
        
        # 按比例随机采样
        if train_ratio < 1.0:
            train_samples = random.sample(train_samples, int(len(train_samples) * train_ratio))
            logger.info(f"训练集采样后样本数: {len(train_samples)}")
        if val_ratio < 1.0:
            val_samples = random.sample(val_samples, int(len(val_samples) * val_ratio))
            logger.info(f"验证集采样后样本数: {len(val_samples)}")
        if test_ratio < 1.0:
            test_samples = random.sample(test_samples, int(len(test_samples) * test_ratio))
            logger.info(f"测试集采样后样本数: {len(test_samples)}")
        
        # 转换为Dataset对象
        self.train_dataset = Dataset.from_list(train_samples)
        self.val_dataset = Dataset.from_list(val_samples)
        self.test_dataset = Dataset.from_list(test_samples)
        
        logger.info(f"数据集准备完成: 训练集={len(self.train_dataset)}, 验证集={len(self.val_dataset)}, 测试集={len(self.test_dataset)}")
        
        # 释放原始DataFrame内存
        del self.val_df, self.test_df
        gc.collect()
        logger.info("已释放验证集和测试集DataFrame内存")
        
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
        """预处理数据集，优化内存使用"""
        logger.info("开始预处理数据集...")
        
        def tokenize_function(examples):
            # 编码输入文本 - 使用动态填充
            inputs = self.tokenizer(
                examples["input_text"], 
                max_length=self.max_length, 
                truncation=True,
                padding="longest"  # 动态填充到批次中的最大长度，而非固定长度
            )
            
            # 编码目标文本
            targets = self.tokenizer(
                examples["target_text"], 
                max_length=64,  # 标题通常较短
                truncation=True,
                padding="longest"
            )
            
            inputs["labels"] = targets["input_ids"]
            return inputs
        
        # 对数据集进行预处理 - 使用batched=True和num_proc=1
        self.train_dataset = self.train_dataset.map(
            tokenize_function, 
            batched=True,
            num_proc=1  # 不使用多进程，避免内存爆炸
        )
        self.val_dataset = self.val_dataset.map(
            tokenize_function, 
            batched=True,
            num_proc=1
        )
        self.test_dataset = self.test_dataset.map(
            tokenize_function, 
            batched=True,
            num_proc=1
        )
        
        # 设置格式以便与PyTorch兼容 - 使用内存映射
        self.train_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'labels'],
            output_all_columns=True
        )
        self.val_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'labels'],
            output_all_columns=True
        )
        self.test_dataset.set_format(
            type='torch', 
            columns=['input_ids', 'attention_mask', 'labels'],
            output_all_columns=True
        )
        
        logger.info("数据预处理完成")
        return self
    
    def train(self, batch_size=1, num_epochs=3, learning_rate=5e-5):
        """微调BART模型，深度优化内存使用"""
        logger.info(f"开始微调BART模型，参数配置: batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
        
        # 定义训练参数 - 内存优化配置
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
            save_total_limit=2,  # 只保存2个最佳模型
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),  # 启用混合精度训练
            gradient_accumulation_steps=4,  # 梯度累积，模拟更大batch_size
            dataloader_num_workers=0,  # 不使用多线程加载数据
            report_to="none",  # 不向TensorBoard等报告，减少开销
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model,
            padding="longest"  # 动态填充
        )
        
        # 定义评估指标 - 完整日志输出
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # 只在CPU上处理，避免显存占用
            predictions = predictions.cpu()
            labels = labels.cpu()
            
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
            
            logger.info(f"当前评估指标: {result}")
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
        
        # 开始训练前清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU内存已清理，当前GPU内存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 开始训练
        trainer.train()
        
        # 保存最佳模型
        best_model_dir = os.path.join(self.output_dir, "best_model")
        trainer.save_model(best_model_dir)
        self.tokenizer.save_pretrained(best_model_dir)
        
        logger.info(f"模型训练完成，最佳模型已保存至 {best_model_dir}")
        return self
    
    def generate_personalized_title(self, user_id, news_id, max_length=64):
        """为特定用户生成个性化新闻标题，优化内存使用"""
        # 获取用户画像
        user_profile = self.user_profiles.get(user_id, {})
        
        # 获取新闻内容
        if news_id not in self.news_id_to_content:
            logger.error(f"新闻ID {news_id} 不存在于 {self.data_dir}")
            return None
        
        news = self.news_id_to_content[news_id]
        logger.info(f"正在为用户 {user_id} 生成新闻 {news_id} 的个性化标题")
        
        # 构建输入文本
        user_pref_text = self._build_user_preference_text(user_profile)
        input_text = f"{user_pref_text} {self.user_token} {news['body']}"
        
        # 编码输入 - 只加载必要部分到GPU
        with torch.no_grad():
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # 生成标题 - 使用低内存生成参数
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=3,  # 减少beam数量
                early_stopping=True,
                no_repeat_ngram_size=2,
                output_scores=False,  # 不输出分数
                return_dict_in_generate=False  # 不返回字典
            )
        
        # 解码生成的标题 - 在CPU上处理
        generated_title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"标题生成完成: {generated_title}")
        
        return {
            'user_id': user_id,
            'news_id': news_id,
            'original_title': news['title'],
            'generated_title': generated_title,
            'user_preferences': user_profile
        }
    
    def evaluate(self, sample_size=50):
        """评估模型性能，优化内存使用"""
        logger.info(f"开始模型评估，评估样本量: {sample_size}")
        
        # 随机选择样本进行评估 - 减少样本量
        if sample_size < len(self.test_dataset):
            indices = random.sample(range(len(self.test_dataset)), sample_size)
            samples = [self.test_dataset[i] for i in indices]
            logger.info(f"随机抽取了 {len(samples)} 个样本进行评估")
        else:
            samples = self.test_dataset
            logger.info(f"使用全部 {len(samples)} 个样本进行评估")
        
        results = []
        for sample in tqdm.tqdm(samples, desc="生成评估结果"):
            user_id = sample['user_id']
            news_id = sample['news_id']
            
            # 生成个性化标题
            result = self.generate_personalized_title(user_id, news_id)
            if result:
                results.append(result)
        
        # 计算评估指标
        if results:
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
            
            logger.info(f"评估完成，ROUGE指标: ROUGE-1={rouge_scores['rouge1']:.2f}, ROUGE-2={rouge_scores['rouge2']:.2f}, ROUGE-L={rouge_scores['rougeL']:.2f}")
            
            # 保存示例结果
            sample_results_path = os.path.join(self.output_dir, "sample_results.txt")
            with open(sample_results_path, "w", encoding="utf-8") as f:
                f.write("个性化新闻标题生成示例:\n\n")
                for i, result in enumerate(results[:5]):  # 只保存5个示例
                    f.write(f"示例 {i+1}:\n")
                    f.write(f"用户ID: {result['user_id']}\n")
                    f.write(f"用户偏好: {result['user_preferences']}\n")
                    f.write(f"原始标题: {result['original_title']}\n")
                    f.write(f"生成标题: {result['generated_title']}\n")
                    f.write("-" * 50 + "\n\n")
            
            return {
                'rouge_scores': rouge_scores,
                'sample_results': results[:5]
            }
        else:
            logger.warning("没有可用的评估结果")
            return {'rouge_scores': {}, 'sample_results': []}

# 主函数
if __name__ == "__main__":
    # 初始化生成器 - 使用base版本模型
    generator = PersonalizedTitleGenerator(model_name="facebook/bart-base")
    
    # 加载数据
    generator.load_data()
    
    # 创建用户画像
    generator.create_user_profiles()
    
    # 准备数据集，使用1%的样本
    generator.prepare_datasets(train_ratio=0.01, val_ratio=0.01, test_ratio=0.01)
    
    # 预处理数据
    generator.preprocess_data()
    
    # 训练模型 - 深度优化内存
    generator.train(batch_size=1, num_epochs=3)
    
    # 评估模型 - 减少评估样本
    evaluation_results = generator.evaluate(sample_size=50)
    
    # 示例：为特定用户生成个性化标题
    if generator.test_dataset and evaluation_results['sample_results']:
        sample_result = evaluation_results['sample_results'][0]
        print("\n个性化标题生成示例:")
        print(f"用户ID: {sample_result['user_id']}")
        print(f"新闻ID: {sample_result['news_id']}")
        print(f"原始标题: {sample_result['original_title']}")
        print(f"生成标题: {sample_result['generated_title']}")
        print(f"用户偏好: {sample_result['user_preferences']}")
    else:
        logger.warning("测试数据集为空或无评估结果，无法展示生成示例")