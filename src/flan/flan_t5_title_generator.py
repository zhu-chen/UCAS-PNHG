import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
import logging
import gc
import tqdm
import random
from collections import defaultdict
import numpy as np
import nltk
from evaluate import load as load_metric
from rouge_score import rouge_scorer
import psutil
from transformers.trainer_callback import TrainerCallback
import json
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

# 设置日志级别为INFO，显示完整中文日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flan_title_generator.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 内存监控工具
def monitor_memory():
    """监控系统内存使用"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024**2  # MB
    logger.info(f"当前内存占用: {mem:.2f} MB")
    return mem

# 内存清理回调
class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:  # 每100步清理一次
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU内存清理后: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

class PersonalizedTitleGenerator:
    def __init__(self, max_length=384, local_model_path=None):
        """初始化个性化标题生成器，从本地加载FLAN-T5模型"""
        self.max_length = max_length
        self.local_model_path = local_model_path

        # UCAS项目目录结构 - FLAN-T5专用路径
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.project_root, "..", "scripts", "data", "raw")
        
        # FLAN-T5结果输出目录
        self.output_dir = os.path.join(self.project_root, "..", "results", "flan")
        os.makedirs(self.output_dir, exist_ok=True)

        # 确保提供了本地模型路径
        if not self.local_model_path:
            self.local_model_path = os.path.abspath(
                os.path.join(self.project_root, "..", "scripts", "local_flan_t5_model")
            )
            
        logger.info(f"从本地路径 {self.local_model_path} 加载FLAN-T5模型和分词器")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.local_model_path,
                local_files_only=True
            )
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.local_model_path,
                local_files_only=True
            )
            logger.info("本地FLAN-T5模型加载成功")
        except Exception as e:
            logger.error(f"本地FLAN-T5模型加载失败: {str(e)}")
            raise ValueError(f"本地模型加载失败，请检查路径 {self.local_model_path}")

        # 设置特殊标记 - FLAN-T5使用不同的tokenizer
        self.user_token = "[user]"
        
        # FLAN-T5的tokenizer不需要添加新token，但确保特殊标记被正确处理
        logger.info(f"使用特殊标记: {self.user_token}")

        # 评估指标设置
        self.setup_evaluation_metrics()

        # 创建输出目录
        logger.info(f"初始化完成，FLAN-T5模型保存路径：{self.output_dir}")

    def setup_evaluation_metrics(self):
        """设置评估指标，支持多种ROUGE实现"""
        self.rouge_available = False
        self.rouge_impl = None

        try:
            # 优先尝试使用evaluate库的ROUGE实现
            self.rouge_metric = load_metric("rouge")
            self.rouge_available = True
            self.rouge_impl = "evaluate"
            logger.info("使用evaluate库的ROUGE实现")
        except Exception as e:
            logger.warning(f"evaluate库ROUGE加载失败: {str(e)}，尝试使用rouge-score")
            try:
                # 备用方案：使用rouge-score
                self.rouge_scorer = rouge_scorer.RougeScorer(
                    ['rouge1', 'rouge2', 'rougeL'], 
                    use_stemmer=True
                )
                self.rouge_available = True
                self.rouge_impl = "rouge-score"
                logger.info("使用rouge-score的ROUGE实现")
            except Exception as e:
                logger.error(f"所有ROUGE实现加载失败: {str(e)}")
                
        # 设置BLEU平滑函数
        self.smoothing_func = SmoothingFunction().method1

    def load_data(self):
        """优化数据加载方式，适应pickle文件特性"""
        logger.info(f"开始从{self.data_dir}加载数据...")
        monitor_memory()
        try:
            news_path = os.path.join(self.data_dir, "news_corpus.pkl")
            train_path = os.path.join(self.data_dir, "train.pkl")
            val_path = os.path.join(self.data_dir, "validation.pkl")
            test_path = os.path.join(self.data_dir, "test.pkl")

            # 读取新闻数据
            self.news_df = pd.read_pickle(news_path, compression='infer')
            monitor_memory()
            
            # 构建新闻ID到内容的映射
            self.news_id_to_content = self.news_df.set_index('news_id')[['title', 'body', 'category']].to_dict(orient='index')
            del self.news_df  # 释放内存
            gc.collect()
            monitor_memory()

            # 读取完整训练数据
            self.train_df = pd.read_pickle(train_path, compression='infer')
            monitor_memory()

            # 同理读取验证和测试数据
            self.val_df = pd.read_pickle(val_path, compression='infer')
            monitor_memory()
            
            self.test_df = pd.read_pickle(test_path, compression='infer')
            monitor_memory()

            logger.info(f"数据加载完成: 训练集{len(self.train_df)}, 验证集{len(self.val_df)}, 测试集{len(self.test_df)}")
            
            # 检查数据格式
            logger.info(f"训练数据列名: {list(self.train_df.columns)}")
            logger.info(f"验证数据列名: {list(self.val_df.columns)}")
            logger.info(f"测试数据列名: {list(self.test_df.columns)}")
            
            # 检查关键列是否存在 - 修复大小写问题
            train_required_columns = ['UserID', 'ClicknewsID']
            val_required_columns = ['UserID', 'ClicknewsID']
            test_required_columns = ['userid', 'clicknewsID']  # 使用小写列名
            
            if not all(col in self.train_df.columns for col in train_required_columns):
                logger.warning(f"训练数据缺少必要的列: {train_required_columns}")
            if not all(col in self.val_df.columns for col in val_required_columns):
                logger.warning(f"验证数据缺少必要的列: {val_required_columns}")
            if not all(col in self.test_df.columns for col in test_required_columns):
                logger.warning(f"测试数据缺少必要的列: {test_required_columns}")
            
            return self

        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            raise

    def create_user_profiles(self):
        """创建稀疏化用户画像，只保留 top N 兴趣"""
        logger.info("开始构建稀疏化用户兴趣画像...")
        monitor_memory()
        try:
            self.user_profiles = defaultdict(dict)
            
            # 分块处理训练数据（使用固定大小的chunksize进行处理）
            chunksize = 10000
            total_chunks = len(self.train_df) // chunksize + 1
            
            for i in tqdm.tqdm(range(0, len(self.train_df), chunksize), total=total_chunks, desc="构建用户画像"):
                chunk = self.train_df.iloc[i:i+chunksize]
                for _, row in chunk.iterrows():
                    user_id = row.get('UserID')
                    if pd.notna(user_id):
                        clicked_news_ids = str(row.get('ClicknewsID', '')).split()
                        for news_id in clicked_news_ids:
                            if news_id in self.news_id_to_content:
                                category = self.news_id_to_content[news_id].get('category', 'unknown')
                                self.user_profiles[user_id][category] = self.user_profiles[user_id].get(category, 0) + 1
                
                # 每处理完一个块，释放内存
                del chunk
                gc.collect()
                
                # 每10个块记录一次进度
                if (i // chunksize) % 10 == 0:
                    logger.info(f"用户画像构建中，已处理 {i // chunksize}/{total_chunks} 块")
                    monitor_memory()
            
            # 转换为百分比并只保留top 5兴趣
            for user_id, profile in self.user_profiles.items():
                if profile:
                    total = sum(profile.values())
                    if total > 0:
                        sorted_prof = sorted(profile.items(), key=lambda x: x[1], reverse=True)[:5]
                        self.user_profiles[user_id] = {cat: cnt/total for cat, cnt in sorted_prof}
                    else:
                        self.user_profiles[user_id] = {}
            
            logger.info(f"稀疏化用户画像构建完成，共{len(self.user_profiles)}个用户")
            
            # 释放不再使用的训练数据
            del self.train_df
            gc.collect()
            monitor_memory()
            return self

        except Exception as e:
            logger.error(f"用户画像构建失败: {str(e)}")
            raise

    def prepare_datasets(self, train_ratio=0.01, val_ratio=0.01, test_ratio=0.01):
        """准备训练、验证和测试数据集，优化内存使用"""
        logger.info(f"开始准备数据集，采样比例: 训练集{train_ratio*100:.1f}%, 验证集{val_ratio*100:.1f}%, 测试集{test_ratio*100:.1f}%")
        monitor_memory()
        try:
            train_samples = []
            val_samples = []
            test_samples = []

            # 分块处理验证数据构建训练样本
            chunksize = 1000
            total_chunks = len(self.val_df) // chunksize + 1
            
            for i in tqdm.tqdm(range(0, len(self.val_df), chunksize), total=total_chunks, desc="准备训练数据"):
                chunk = self.val_df.iloc[i:i+chunksize]
                for _, row in chunk.iterrows():
                    user_id = row.get('UserID')
                    if pd.notna(user_id):
                        user_profile = self.user_profiles.get(user_id, {})
                        clicked_news_ids = str(row.get('ClicknewsID', '')).split()
                        for news_id in clicked_news_ids:
                            if news_id and news_id in self.news_id_to_content:
                                news = self.news_id_to_content[news_id]
                                user_pref_text = self._build_user_preference_text(user_profile)
                                input_text = f"{user_pref_text} {self.user_token} {news['body']}"
                                train_samples.append({
                                    'user_id': user_id,
                                    'news_id': news_id,
                                    'input_text': input_text,
                                    'target_text': news['title']
                                })
                
                # 每处理完一个块，释放内存
                del chunk
                gc.collect()
                
                # 每10个块记录一次进度
                if (i // chunksize) % 10 == 0:
                    logger.info(f"训练数据准备中，已处理 {i // chunksize}/{total_chunks} 块，样本数: {len(train_samples)}")
                    monitor_memory()
                    
                    # 如果已达到采样量，停止处理
                    if train_ratio < 1.0 and len(train_samples) >= int(len(self.val_df) * train_ratio):
                        break
            
            # 同理处理验证样本
            total_chunks = len(self.val_df) // chunksize + 1
            for i in tqdm.tqdm(range(0, len(self.val_df), chunksize), total=total_chunks, desc="准备验证数据"):
                chunk = self.val_df.iloc[i:i+chunksize]
                for _, row in chunk.iterrows():
                    user_id = row.get('UserID')
                    if pd.notna(user_id):
                        user_profile = self.user_profiles.get(user_id, {})
                        clicked_news_ids = str(row.get('ClicknewsID', '')).split()
                        for news_id in clicked_news_ids:
                            if news_id and news_id in self.news_id_to_content:
                                news = self.news_id_to_content[news_id]
                                user_pref_text = self._build_user_preference_text(user_profile)
                                input_text = f"{user_pref_text} {self.user_token} {news['body']}"
                                val_samples.append({
                                    'user_id': user_id,
                                    'news_id': news_id,
                                    'input_text': input_text,
                                    'target_text': news['title']
                                })
                
                # 每处理完一个块，释放内存
                del chunk
                gc.collect()
                
                # 每10个块记录一次进度
                if (i // chunksize) % 10 == 0:
                    logger.info(f"验证数据准备中，已处理 {i // chunksize}/{total_chunks} 块，样本数: {len(val_samples)}")
                    monitor_memory()
                    
                    if val_ratio < 1.0 and len(val_samples) >= int(len(self.val_df) * val_ratio):
                        break
            
            # 处理测试样本 - 使用小写列名 'userid' 和 'clicknewsID'
            total_chunks = len(self.test_df) // chunksize + 1
            for i in tqdm.tqdm(range(0, len(self.test_df), chunksize), total=total_chunks, desc="准备测试数据"):
                chunk = self.test_df.iloc[i:i+chunksize]
                for _, row in chunk.iterrows():
                    # 使用小写列名 'userid' 和 'clicknewsID'
                    user_id = row.get('userid', '')
                    if pd.notna(user_id):
                        user_profile = self.user_profiles.get(user_id, {})
                        clicked_news_ids = str(row.get('clicknewsID', '')).split(',')
                        for news_id in clicked_news_ids:
                            if news_id and news_id in self.news_id_to_content:
                                news = self.news_id_to_content[news_id]
                                user_pref_text = self._build_user_preference_text(user_profile)
                                input_text = f"{user_pref_text} {self.user_token} {news['body']}"
                                test_samples.append({
                                    'user_id': user_id,
                                    'news_id': news_id,
                                    'input_text': input_text,
                                    'target_text': news['title']
                                })
                
                # 每处理完一个块，释放内存
                del chunk
                gc.collect()
                
                # 每10个块记录一次进度
                if (i // chunksize) % 10 == 0:
                    logger.info(f"测试数据准备中，已处理 {i // chunksize}/{total_chunks} 块，样本数: {len(test_samples)}")
                    monitor_memory()
                    
                    # 测试集采样到与训练集相当的数量
                    target_size = min(len(self.val_df) * train_ratio, 600)  # 不超过600
                    if test_ratio < 1.0 and len(test_samples) >= target_size:
                        break
            
            # 释放不再使用的数据
            del self.val_df, self.test_df
            gc.collect()
            monitor_memory()

            # 检查样本数量
            logger.info(f"准备完成: 训练样本{len(train_samples)}, 验证样本{len(val_samples)}, 测试样本{len(test_samples)}")
            
            # 如果任何数据集为空，记录警告
            if len(train_samples) == 0:
                logger.warning("训练样本为空！请检查数据和采样比例")
            if len(val_samples) == 0:
                logger.warning("验证样本为空！请检查数据和采样比例")
            if len(test_samples) == 0:
                logger.warning("测试样本为空！请检查数据和采样比例")
            
            # 按比例采样（如果在处理过程中未达到采样量）
            if train_ratio < 1.0 and len(train_samples) > int(len(train_samples) * train_ratio):
                train_samples = random.sample(train_samples, int(len(train_samples) * train_ratio))
                logger.info(f"训练集采样后样本数: {len(train_samples)}")
            if val_ratio < 1.0 and len(val_samples) > int(len(val_samples) * val_ratio):
                val_samples = random.sample(val_samples, int(len(val_samples) * val_ratio))
                logger.info(f"验证集采样后样本数: {len(val_samples)}")
            if test_ratio < 1.0 and len(test_samples) > int(len(test_samples) * test_ratio):
                target_size = min(len(test_samples) * test_ratio, 600)  # 不超过600
                test_samples = random.sample(test_samples, int(target_size))
                logger.info(f"测试集采样后样本数: {len(test_samples)}")

            # 检查采样后样本数量
            logger.info(f"采样后: 训练样本{len(train_samples)}, 验证样本{len(val_samples)}, 测试样本{len(test_samples)}")
            
            # 如果任何数据集为空，记录错误
            if len(train_samples) == 0:
                logger.error("训练样本采样后为空！请调整采样比例")
            if len(val_samples) == 0:
                logger.error("验证样本采样后为空！请调整采样比例")
            if len(test_samples) == 0:
                logger.error("测试样本采样后为空！请调整采样比例")
            
            # 显示数据集列名（用于调试）
            if train_samples:
                logger.info(f"训练集列名: {list(train_samples[0].keys())}")
            if val_samples:
                logger.info(f"验证集列名: {list(val_samples[0].keys())}")
            if test_samples:
                logger.info(f"测试集列名: {list(test_samples[0].keys())}")

            # 转换为Dataset对象
            self.train_dataset = Dataset.from_list(train_samples) if train_samples else Dataset.from_dict({})
            self.val_dataset = Dataset.from_list(val_samples) if val_samples else Dataset.from_dict({})
            self.test_dataset = Dataset.from_list(test_samples) if test_samples else Dataset.from_dict({})

            logger.info(f"数据集准备完成: 训练集{len(self.train_dataset)}, 验证集{len(self.val_dataset)}, 测试集{len(self.test_dataset)}")
            
            # 显示转换后的数据集列名（用于调试）
            if len(self.train_dataset) > 0:
                logger.info(f"训练数据集列名: {self.train_dataset.column_names}")
            if len(self.val_dataset) > 0:
                logger.info(f"验证数据集列名: {self.val_dataset.column_names}")
            if len(self.test_dataset) > 0:
                logger.info(f"测试数据集列名: {self.test_dataset.column_names}")
            
            # 检查数据集是否包含必要的列
            required_columns = ['input_text', 'target_text']
            if len(self.train_dataset) > 0 and not all(col in self.train_dataset.column_names for col in required_columns):
                logger.error(f"训练数据集缺少必要的列: {required_columns}")
            if len(self.val_dataset) > 0 and not all(col in self.val_dataset.column_names for col in required_columns):
                logger.error(f"验证数据集缺少必要的列: {required_columns}")
            if len(self.test_dataset) > 0 and not all(col in self.test_dataset.column_names for col in required_columns):
                logger.error(f"测试数据集缺少必要的列: {required_columns}")

            del train_samples, val_samples, test_samples
            gc.collect()
            monitor_memory()
            return self

        except Exception as e:
            logger.error(f"数据集准备失败: {str(e)}")
            raise

    def _build_user_preference_text(self, user_profile):
        """构建用户偏好文本表示"""
        if not user_profile:
            return "用户兴趣: 无明确偏好"

        sorted_preferences = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)
        top_preferences = sorted_preferences[:3]

        pref_text = "用户兴趣: "
        for category, score in top_preferences:
            pref_text += f"{category}({int(score * 100)}%), "
        return pref_text.rstrip(', ')

    def preprocess_data(self):
        """预处理数据集，优化内存使用"""
        logger.info("开始预处理数据集...")
        monitor_memory()
        try:
            def tokenize_function(examples):
                # FLAN-T5使用与BART不同的输入格式
                inputs = self.tokenizer(
                    examples["input_text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="longest"
                )
                # FLAN-T5的目标文本需要单独处理
                with self.tokenizer.as_target_tokenizer():
                    targets = self.tokenizer(
                        examples["target_text"],
                        max_length=64,
                        truncation=True,
                        padding="longest"
                    )
                inputs["labels"] = targets["input_ids"]
                return inputs

            # 分批次处理，减少内存峰值
            batch_size = 1000
            
            # 检查数据集是否包含必要的列
            required_columns = ['input_text', 'target_text']
            
            # 处理训练集
            if len(self.train_dataset) > 0 and all(col in self.train_dataset.column_names for col in required_columns):
                self.train_dataset = self.train_dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=1,
                    remove_columns=['input_text', 'target_text']  # 移除原始文本减少内存
                )
            elif len(self.train_dataset) > 0:
                logger.warning(f"训练数据集缺少必要的列，将不进行列移除操作。当前列: {self.train_dataset.column_names}")
                self.train_dataset = self.train_dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=1
                )
            
            monitor_memory()

            # 处理验证集
            if len(self.val_dataset) > 0 and all(col in self.val_dataset.column_names for col in required_columns):
                self.val_dataset = self.val_dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=1,
                    remove_columns=['input_text', 'target_text']
                )
            elif len(self.val_dataset) > 0:
                logger.warning(f"验证数据集缺少必要的列，将不进行列移除操作。当前列: {self.val_dataset.column_names}")
                self.val_dataset = self.val_dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=1
                )
            
            monitor_memory()

            # 处理测试集
            if len(self.test_dataset) > 0 and all(col in self.test_dataset.column_names for col in required_columns):
                self.test_dataset = self.test_dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=1,
                    remove_columns=['input_text', 'target_text']
                )
            elif len(self.test_dataset) > 0:
                logger.warning(f"测试数据集缺少必要的列，将不进行列移除操作。当前列: {self.test_dataset.column_names}")
                self.test_dataset = self.test_dataset.map(
                    tokenize_function,
                    batched=True,
                    batch_size=batch_size,
                    num_proc=1
                )
            
            monitor_memory()

            # 设置为torch格式
            if len(self.train_dataset) > 0:
                self.train_dataset.set_format(
                    type='torch',
                    columns=['input_ids', 'attention_mask', 'labels'],
                    output_all_columns=True
                )
            if len(self.val_dataset) > 0:
                self.val_dataset.set_format(
                    type='torch',
                    columns=['input_ids', 'attention_mask', 'labels'],
                    output_all_columns=True
                )
            if len(self.test_dataset) > 0:
                self.test_dataset.set_format(
                    type='torch',
                    columns=['input_ids', 'attention_mask', 'labels'],
                    output_all_columns=True
                )

            logger.info("数据预处理完成")
            monitor_memory()
            return self

        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def train(self, batch_size=1, num_epochs=3, learning_rate=5e-5):
        """深度优化内存的训练流程"""
        logger.info(f"开始内存优化模式训练，batch_size={batch_size}, epochs={num_epochs}")
        monitor_memory()
        
        # 预清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"初始GPU内存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 低内存训练参数 - 进一步降低内存消耗
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,  # 最小batch_size
            per_device_eval_batch_size=1,  # 评估时使用更小的batch_size
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            save_total_limit=1,  # 只保存最新模型
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  # 使用损失作为评估指标
            greater_is_better=False,  # 损失越小越好
            fp16=True,  # 强制使用混合精度
            gradient_accumulation_steps=16,  # 增加累积步数，模拟大batch
            dataloader_num_workers=0,  # 不使用多线程
            report_to="none",
            # 新增内存优化参数
            torch_compile=False,  # 关闭编译优化
            no_cuda=not torch.cuda.is_available(),  # 强制使用CPU（如果GPU内存不足）
            eval_accumulation_steps=2,  # 减少评估时的内存峰值
        )
        
        # 数据收集器（动态填充）
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
            max_length=self.max_length,
        )
        
        # 简化的评估指标函数
        def compute_metrics(eval_pred):
            """简化的评估指标，仅用于监控训练过程"""
            # 这里不再尝试计算ROUGE，以避免错误
            return {}
        
        # 创建训练器（含内存清理回调）
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset if len(self.train_dataset) > 0 else None,
            eval_dataset=self.val_dataset if len(self.val_dataset) > 0 else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics,  # 使用简化的评估函数
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=2),
                MemoryCleanupCallback()
            ]
        )
        
        # 执行训练
        try:
            if len(self.train_dataset) > 0:
                trainer.train()
                logger.info("训练完成，清理最终内存")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"最终GPU内存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            else:
                logger.error("没有可用的训练数据，跳过训练")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("训练过程中内存溢出，请进一步降低batch_size或采样比例")
                # 建议降低采样比例或增加梯度累积步数
                logger.info("建议解决方案：降低采样比例或增加梯度累积步数")
            raise
        
        # 保存模型
        best_model_dir = os.path.join(self.output_dir, "best_model")
        if len(self.train_dataset) > 0:
            trainer.save_model(best_model_dir)
            self.tokenizer.save_pretrained(best_model_dir)
            monitor_memory()
            logger.info(f"FLAN-T5模型保存至 {best_model_dir}")
        
        return self

    def generate_personalized_title(self, user_id, news_id, max_length=64):
        """为特定用户生成个性化新闻标题"""
        logger.info(f"为用户{user_id}生成新闻{news_id}的标题")
        monitor_memory()
        try:
            user_profile = self.user_profiles.get(user_id, {})
            if news_id not in self.news_id_to_content:
                logger.error(f"新闻ID{news_id}不存在")
                return None

            news = self.news_id_to_content[news_id]
            user_pref_text = self._build_user_preference_text(user_profile)
            input_text = f"{user_pref_text} {self.user_token} {news['body']}"

            with torch.no_grad():
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                # FLAN-T5生成配置
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_beams=3,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    output_scores=False,
                    return_dict_in_generate=False
                )

            generated_title = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"标题生成完成: {generated_title}")
            monitor_memory()

            return {
                'user_id': user_id,
                'news_id': news_id,
                'original_title': news['title'],
                'generated_title': generated_title,
                'user_preferences': user_profile
            }

        except Exception as e:
            logger.error(f"标题生成失败: {str(e)}")
            return None

    def evaluate(self, sample_size=50):
        """评估模型性能"""
        logger.info(f"开始模型评估，样本量: {sample_size}")
        monitor_memory()
        try:
            if not hasattr(self, 'test_dataset') or len(self.test_dataset) == 0:
                logger.warning("测试数据集为空")
                return {'rouge_scores': {}, 'bleu_scores': {}, 'sample_results': []}

            # 确保样本量不超过测试集大小
            sample_size = min(sample_size, len(self.test_dataset))
            
            indices = random.sample(range(len(self.test_dataset)), sample_size)
            samples = [self.test_dataset[i] for i in indices]
            logger.info(f"随机抽取{len(samples)}个样本评估")
            
            # 分批次生成结果，减少内存占用
            results = []
            batch_size = 5  # 减小批大小以减少内存使用
            for i in tqdm.tqdm(range(0, len(samples), batch_size), desc="生成评估结果"):
                batch = samples[i:i+batch_size]
                for sample in batch:
                    user_id = sample.get('user_id', '')
                    news_id = sample.get('news_id', '')
                    if user_id and news_id:
                        result = self.generate_personalized_title(user_id, news_id)
                        if result:
                            results.append(result)
                
                # 每批次后清理内存
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            monitor_memory()

            if not results:
                logger.warning("没有生成有效评估结果")
                return {'rouge_scores': {}, 'bleu_scores': {}, 'sample_results': []}

            references = [r['original_title'] for r in results]
            predictions = [r['generated_title'] for r in results]

            # 计算ROUGE指标
            if self.rouge_impl == "evaluate":
                rouge_scores = self.rouge_metric.compute(
                    predictions=predictions,
                    references=references,
                    use_stemmer=True
                )
                formatted_rouge = {
                    "rouge1": rouge_scores["rouge1"].mid.fmeasure * 100,
                    "rouge2": rouge_scores["rouge2"].mid.fmeasure * 100,
                    "rougeL": rouge_scores["rougeL"].mid.fmeasure * 100
                }
            elif self.rouge_impl == "rouge-score":
                scores = [self.rouge_scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
                formatted_rouge = {
                    "rouge1": np.mean([s["rouge1"].fmeasure for s in scores]) * 100,
                    "rouge2": np.mean([s["rouge2"].fmeasure for s in scores]) * 100,
                    "rougeL": np.mean([s["rougeL"].fmeasure for s in scores]) * 100
                }
            else:
                formatted_rouge = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
                logger.warning("ROUGE实现不可用，跳过ROUGE计算")

            # 计算BLEU指标
            bleu_scores = []
            for ref, pred in zip(references, predictions):
                # 将中文句子转换为字符列表
                ref_chars = list(ref)
                pred_chars = list(pred)
                
                # 计算BLEU分数（使用平滑函数处理短句子）
                bleu = sentence_bleu([ref_chars], pred_chars, 
                                     smoothing_function=self.smoothing_func)
                bleu_scores.append(bleu)
            
            avg_bleu = np.mean(bleu_scores) * 100
            formatted_bleu = {"bleu": avg_bleu}
            
            # 将指标添加到结果中
            for i, r in enumerate(results):
                r['metrics'] = {
                    'rouge1': formatted_rouge['rouge1'] if i == 0 else 0,  # 只记录一次全局指标
                    'rouge2': formatted_rouge['rouge2'] if i == 0 else 0,
                    'rougeL': formatted_rouge['rougeL'] if i == 0 else 0,
                    'bleu': bleu_scores[i] * 100
                }

            logger.info(f"评估完成，ROUGE指标: ROUGE-1={formatted_rouge['rouge1']:.2f}, ROUGE-2={formatted_rouge['rouge2']:.2f}, ROUGE-L={formatted_rouge['rougeL']:.2f}")
            logger.info(f"BLEU指标: {avg_bleu:.2f}")
            
            # 保存评估结果
            sample_results_path = os.path.join(self.output_dir, "sample_results.txt")
            with open(sample_results_path, "w", encoding="utf-8") as f:
                f.write("个性化新闻标题生成示例 (FLAN-T5):\n\n")
                f.write(f"整体指标: ROUGE-1={formatted_rouge['rouge1']:.2f}, ROUGE-2={formatted_rouge['rouge2']:.2f}, ROUGE-L={formatted_rouge['rougeL']:.2f}, BLEU={avg_bleu:.2f}\n\n")
                
                for i, result in enumerate(results[:5]):
                    f.write(f"示例 {i+1}:\n")
                    f.write(f"用户ID: {result['user_id']}\n")
                    f.write(f"用户偏好: {result['user_preferences']}\n")
                    f.write(f"原始标题: {result['original_title']}\n")
                    f.write(f"生成标题: {result['generated_title']}\n")
                    
                    # 添加指标信息
                    if 'metrics' in result:
                        metrics = result['metrics']
                        f.write(f"指标: ROUGE-1={metrics['rouge1']:.2f}, ROUGE-2={metrics['rouge2']:.2f}, ROUGE-L={metrics['rougeL']:.2f}, BLEU={metrics['bleu']:.2f}\n")
                    
                    f.write("-" * 50 + "\n\n")
            
            # 保存详细结果到JSON
            detailed_results_path = os.path.join(self.output_dir, "detailed_results.json")
            with open(detailed_results_path, "w", encoding="utf-8") as f:
                json.dump({
                    'overall_metrics': {
                        'rouge': formatted_rouge,
                        'bleu': formatted_bleu
                    },
                    'sample_results': results[:10]  # 保存前10个样本的详细结果
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"详细评估结果已保存至: {detailed_results_path}")
            
            monitor_memory()
            return {
                'rouge_scores': formatted_rouge,
                'bleu_scores': formatted_bleu,
                'sample_results': results[:5]
            }

        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            return {'rouge_scores': {}, 'bleu_scores': {}, 'sample_results': []}

if __name__ == "__main__":
    try:
        # 确保nltk资源可用
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # 添加超时和重试机制
        import socket
        socket.setdefaulttimeout(10)  # 设置10秒超时
        try:
            nltk.download('punkt', quiet=True)
            logger.info("已下载nltk punkt资源")
        except Exception as e:
            logger.error(f"下载nltk资源失败: {str(e)}，但程序将继续运行")
    
    # 设置本地模型路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_model_path = os.path.abspath(
        os.path.join(project_root, "..", "scripts", "local_flan_t5_model")
    )
    
    # 初始化FLAN-T5生成器 - 只从本地加载
    generator = PersonalizedTitleGenerator(
        local_model_path=local_model_path
    )
    
    # 执行流程 - 降低采样比例以减少内存使用
    if generator.load_data():
        if generator.create_user_profiles():
            # 使用较小的采样比例进行测试
            if generator.prepare_datasets(train_ratio=0.005, val_ratio=0.005, test_ratio=0.05):  # 降低测试集比例
                if generator.preprocess_data():
                    # 使用最小batch_size和最大梯度累积
                    generator.train(batch_size=1, num_epochs=1, learning_rate=5e-5)
                    
                    # 评估时使用较小的样本量
                    evaluation_results = generator.evaluate(sample_size=50)  # 限制评估样本量
                    
                    if evaluation_results['sample_results']:
                        sample_result = evaluation_results['sample_results'][0]
                        print("\n个性化标题生成示例 (FLAN-T5):")
                        print(f"用户ID: {sample_result['user_id']}")
                        print(f"新闻ID: {sample_result['news_id']}")
                        print(f"原始标题: {sample_result['original_title']}")
                        print(f"生成标题: {sample_result['generated_title']}")
                        print(f"用户偏好: {sample_result['user_preferences']}")
                        
                        # 打印整体指标
                        print("\n整体评估指标 (FLAN-T5):")
                        print(f"ROUGE-1: {evaluation_results['rouge_scores']['rouge1']:.2f}")
                        print(f"ROUGE-2: {evaluation_results['rouge_scores']['rouge2']:.2f}")
                        print(f"ROUGE-L: {evaluation_results['rouge_scores']['rougeL']:.2f}")
                        print(f"BLEU: {evaluation_results['bleu_scores']['bleu']:.2f}")
                    else:
                        logger.warning("无有效评估结果用于展示")
                else:
                    logger.error("数据预处理失败")
            else:
                logger.error("数据集准备失败")
        else:
            logger.error("用户画像构建失败")
    else:
        logger.error("数据加载失败")