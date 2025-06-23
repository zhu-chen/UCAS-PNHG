"""
PENS数据预处理器
处理原始PENS数据集，进行清洗和格式化
"""

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
from collections import defaultdict
import os

logger = logging.getLogger(__name__)


class PENSPreprocessor:
    """PENS数据预处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 预处理配置
        """
        self.config = config
        self.category_mapping = {}
        self.user_mapping = {}
        self.news_corpus = {}  # 新闻ID到内容的映射
        
    def preprocess_raw_data(
        self, 
        raw_data_dir: str, 
        output_dir: str
    ) -> Tuple[str, str, str]:
        """
        预处理原始PENS数据 - 分块处理版本
        
        Args:
            raw_data_dir: 原始数据目录
            output_dir: 输出目录
            
        Returns:
            训练集、验证集、测试集路径
        """
        raw_dir = Path(raw_data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 首先加载新闻库
        self._load_news_corpus(raw_dir)
        
        # 加载原始数据
        train_df = self._load_impression_file(raw_dir / "train.pkl")
        valid_df = self._load_impression_file(raw_dir / "validation.pkl") 
        test_df = self._load_test_file(raw_dir / "test.pkl")
        
        logger.info(f"原始数据大小 - 训练: {len(train_df)}, 验证: {len(valid_df)}, 测试: {len(test_df)}")
        
        # 分块处理训练集
        train_path = output_dir / "train_processed.pkl"
        self._preprocess_and_save_in_chunks(train_df, 'train', train_path)
        
        # 分块处理验证集
        valid_path = output_dir / "valid_processed.pkl"
        self._preprocess_and_save_in_chunks(valid_df, 'valid', valid_path)
        
        # 预处理测试集（通常较小，不需要分块）
        test_processed = self._preprocess_personalized_test_data(test_df)
        test_path = output_dir / "test_processed.pkl"
        test_processed.to_pickle(test_path)
        
        # 保存数据统计信息
        stats = self.analyze_data_statistics([str(train_path), str(valid_path), str(test_path)])
        with open(output_dir / "data_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"预处理完成")
        
        return str(train_path), str(valid_path), str(test_path)
    
    def _load_news_corpus(self, raw_dir: Path):
        """加载新闻库"""
        news_corpus_file = raw_dir / "news_corpus.pkl"
        if news_corpus_file.exists():
            news_df = pd.read_pickle(news_corpus_file)
            logger.info(f"加载新闻库，包含 {len(news_df)} 篇文章")
            print(f"新闻库列名: {list(news_df.columns)}")
        else:
            logger.error("未找到新闻库文件 news_corpus.pkl")
            raise FileNotFoundError("新闻库文件不存在")
        
        # 构建新闻ID到内容的映射
        for _, row in news_df.iterrows():
            self.news_corpus[row['news_id']] = {
                'title': self._clean_text(str(row.get('title', ''))),
                'body': self._clean_text(str(row.get('body', ''))),
                'category': row.get('category', 'unknown')
            }
        
        logger.info(f"构建新闻库映射，包含 {len(self.news_corpus)} 篇文章")
    
    def _load_impression_file(self, file_path: Path) -> pd.DataFrame:
        """加载impression数据文件"""
        try:
            if file_path.suffix == '.pkl':
                df = pd.read_pickle(file_path)
            elif file_path.suffix == '.tsv':
                df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
            logger.info(f"加载文件 {file_path.name}，包含 {len(df)} 条记录")
            logger.info(f"文件列名: {list(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"加载文件失败 {file_path}: {e}")
            return pd.DataFrame()
    
    def _load_test_file(self, file_path: Path) -> pd.DataFrame:
        """加载测试数据文件"""
        return self._load_impression_file(file_path)
    
    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 移除特殊字符，保留字母、数字、空格和基本标点
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 去除首尾空格
        text = text.strip()
        
        return text
    
    def _preprocess_impression_data(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """预处理impression数据"""
        processed_data = []
        
        # 添加数据采样以减少内存使用
        sample_ratio = self.config.get('data_sample_ratio', 1.0)  # 默认使用全量数据
        if sample_ratio < 1.0:
            original_len = len(df)
            df = df.sample(frac=sample_ratio, random_state=42)
            logger.info(f"数据采样: 从 {original_len} 条记录采样到 {len(df)} 条 (比例: {sample_ratio})")
        
        logger.info(f"预处理{split}数据，处理数据包含 {len(df)} 条记录...")
        
        processed_count = 0
        max_samples = self.config.get('max_samples_per_split')
        if max_samples is None:
            max_samples = len(df)  # 如果没有限制则处理全部数据
        
        for idx, row in df.iterrows():
            if processed_count >= max_samples:
                logger.info(f"达到最大样本数限制 {max_samples}，停止处理")
                break
                
            if idx % 10000 == 0:  # 增加进度报告频率
                print(f"处理进度: {processed_count}/{min(len(df), max_samples)}")
                
            user_id = str(row.get('UserID', ''))
            
            # 构建用户历史 - 从点击历史中提取
            user_history = self._build_user_history_from_row(row)
            
            # 处理正面新闻（点击的）
            pos_news_str = str(row.get('pos', ''))
            if pos_news_str and pos_news_str != 'nan':
                pos_news_ids = pos_news_str.split()
                
                # 限制每个用户的正面样本数量
                max_pos_per_user = self.config.get('max_pos_samples_per_user', 10)
                pos_news_ids = pos_news_ids[:max_pos_per_user]
                
                for news_id in pos_news_ids:
                    if news_id in self.news_corpus:
                        news_info = self.news_corpus[news_id]
                        
                        # 跳过无效数据
                        if not news_info['title'] or len(news_info['title'].split()) < 3:
                            continue
                        
                        processed_row = {
                            'user_id': user_id,
                            'news_id': news_id,
                            'title': news_info['title'],
                            'body': news_info['body'][:self.config.get('max_body_length', 500)],
                            'category': news_info['category'],
                            'user_history': json.dumps(user_history[:self.config.get('max_user_history', 50)]) if user_history else "[]",
                            'split': split,
                            'label': 1  # 正面样本
                        }
                        
                        processed_data.append(processed_row)
                        processed_count += 1
                        
                        if processed_count >= max_samples:
                            break
            
            if processed_count >= max_samples:
                break
            
            # 处理负面新闻（未点击的）
            neg_news_str = str(row.get('neg', ''))
            if neg_news_str and neg_news_str != 'nan' and self.config.get('include_negative_samples', True):
                neg_news_ids = neg_news_str.split()
                
                # 随机采样负面样本
                neg_sample_ratio = self.config.get('negative_sample_ratio', 0.1)
                max_neg_samples = max(1, int(len(neg_news_ids) * neg_sample_ratio))
                max_neg_per_user = self.config.get('max_neg_samples_per_user', 5)
                max_neg_samples = min(max_neg_samples, max_neg_per_user)
                
                import random
                neg_news_ids = random.sample(neg_news_ids, min(len(neg_news_ids), max_neg_samples))
                
                for news_id in neg_news_ids:
                    if news_id in self.news_corpus:
                        news_info = self.news_corpus[news_id]
                        
                        if not news_info['title'] or len(news_info['title'].split()) < 3:
                            continue
                        
                        processed_row = {
                            'user_id': user_id,
                            'news_id': news_id,
                            'title': news_info['title'],
                            'body': news_info['body'][:self.config.get('max_body_length', 500)],
                            'category': news_info['category'],
                            'user_history': json.dumps(user_history[:self.config.get('max_user_history', 50)]) if user_history else "[]",
                            'split': split,
                            'label': 0  # 负面样本
                        }
                        
                        processed_data.append(processed_row)
                        processed_count += 1
                        
                        if processed_count >= max_samples:
                            break
            
            if processed_count >= max_samples:
                break
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"{split}数据预处理完成，输出 {len(result_df)} 条记录")
        return result_df
    
    def _build_user_history_from_row(self, row) -> List[Dict]:
        """从单行impression数据构建用户历史"""
        history = []
        
        # 从用户的点击历史中提取新闻
        clicked_news_str = str(row.get('ClicknewsID', ''))
        if clicked_news_str and clicked_news_str != 'nan':
            clicked_news_ids = clicked_news_str.split()
            
            # 获取停留时间信息
            dwell_times = []
            if 'dwelltime' in row and pd.notna(row['dwelltime']):
                dwell_times = str(row['dwelltime']).split()
            
            for i, news_id in enumerate(clicked_news_ids):
                if news_id in self.news_corpus:
                    news_info = self.news_corpus[news_id].copy()
                    news_info['news_id'] = news_id
                    
                    # 添加停留时间信息
                    if i < len(dwell_times) and dwell_times[i].isdigit():
                        news_info['dwell_time'] = int(dwell_times[i])
                    else:
                        news_info['dwell_time'] = 0
                    
                    history.append(news_info)
        
        # 限制历史长度
        max_history = self.config.get('max_user_history', 50)
        history = history[-max_history:]
        
        return history
    
    def _preprocess_personalized_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理个性化测试数据"""
        processed_data = []
        
        logger.info(f"预处理个性化测试数据，包含 {len(df)} 条记录...")
        
        for _, row in df.iterrows():
            user_id = str(row.get('userid', ''))
            
            # 获取候选新闻ID
            pos_news_str = str(row.get('posnewID', ''))
            if pos_news_str and pos_news_str != 'nan':
                pos_news_ids = pos_news_str.split(',')
                
                # 获取重写的标题
                rewritten_titles_str = str(row.get('rewrite_titles', ''))
                rewritten_titles = []
                if rewritten_titles_str and rewritten_titles_str != 'nan':
                    rewritten_titles = rewritten_titles_str.split(';;')
                
                # 构建用户历史
                user_history = self._build_test_user_history(row)
                
                for i, news_id in enumerate(pos_news_ids):
                    if news_id in self.news_corpus:
                        news_info = self.news_corpus[news_id]
                        
                        if not news_info['title'] or len(news_info['title'].split()) < 3:
                            continue
                        
                        # 获取对应的个性化标题
                        personalized_title = ""
                        if i < len(rewritten_titles):
                            personalized_title = self._clean_text(rewritten_titles[i])
                        
                        processed_row = {
                            'user_id': user_id,
                            'news_id': news_id,
                            'title': news_info['title'],
                            'body': news_info['body'],
                            'category': news_info['category'],
                            'personalized_title': personalized_title if personalized_title else news_info['title'],
                            'user_history': json.dumps(user_history) if user_history else "[]",
                            'split': 'test'
                        }
                        
                        processed_data.append(processed_row)
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"测试数据预处理完成，输出 {len(result_df)} 条记录")
        return result_df
    
    def _build_test_user_history(self, row) -> List[Dict]:
        """构建测试数据的用户历史"""
        history = []
        
        # 从点击历史中构建
        clicked_news_str = str(row.get('clicknewsID', ''))
        if clicked_news_str and clicked_news_str != 'nan':
            clicked_news_ids = clicked_news_str.split(',')
            
            for news_id in clicked_news_ids:
                if news_id in self.news_corpus:
                    news_info = self.news_corpus[news_id].copy()
                    news_info['news_id'] = news_id
                    history.append(news_info)
        
        # 限制历史长度
        max_history = self.config.get('max_user_history', 50)
        history = history[-max_history:]
        
        return history
    
    def analyze_data_statistics(self, data_paths: List[str]) -> Dict[str, Any]:
        """分析数据统计信息"""
        stats = {
            'total_samples': 0,
            'total_users': set(),
            'total_news': set(),
            'avg_title_length': [],
            'avg_body_length': [],
            'avg_history_length': [],
            'category_distribution': defaultdict(int),
            'splits': {}
        }
        
        for data_path in data_paths:
            split_name = Path(data_path).stem.replace('_processed', '')
            df = pd.read_pickle(data_path)
            
            split_stats = {
                'samples': len(df),
                'users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
                'news': df['news_id'].nunique() if 'news_id' in df.columns else 0
            }
            
            stats['splits'][split_name] = split_stats
            stats['total_samples'] += len(df)
            
            for _, row in df.iterrows():
                if 'user_id' in row:
                    stats['total_users'].add(row['user_id'])
                if 'news_id' in row:
                    stats['total_news'].add(row['news_id'])
                
                # 文本长度统计
                if 'title' in row:
                    title_len = len(str(row['title']).split())
                    stats['avg_title_length'].append(title_len)
                    
                if 'body' in row:
                    body_len = len(str(row['body']).split())
                    stats['avg_body_length'].append(body_len)
                
                # 用户历史长度
                if 'user_history' in row:
                    try:
                        history = json.loads(row['user_history'])
                        stats['avg_history_length'].append(len(history))
                    except:
                        stats['avg_history_length'].append(0)
                
                # 类别分布
                if 'category' in row:
                    stats['category_distribution'][row['category']] += 1
        
        # 计算平均值
        stats['total_users'] = len(stats['total_users'])
        stats['total_news'] = len(stats['total_news'])
        stats['avg_title_length'] = np.mean(stats['avg_title_length']) if stats['avg_title_length'] else 0
        stats['avg_body_length'] = np.mean(stats['avg_body_length']) if stats['avg_body_length'] else 0
        stats['avg_history_length'] = np.mean(stats['avg_history_length']) if stats['avg_history_length'] else 0
        stats['category_distribution'] = dict(stats['category_distribution'])
        
        return stats
    
    def _preprocess_and_save_in_chunks(self, df: pd.DataFrame, split: str, output_path: Path):
        """分块处理并保存数据"""
        chunk_size = self.config.get('chunk_size', 10000)  # 每块处理1万条记录
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        logger.info(f"开始分块处理{split}数据，总共{total_chunks}块，每块{chunk_size}条记录")
        
        # 删除已存在的输出文件
        if output_path.exists():
            output_path.unlink()
        
        all_processed_data = []
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(df))
            
            logger.info(f"处理{split}数据块 {chunk_idx + 1}/{total_chunks} (行 {start_idx}-{end_idx})")
            
            # 获取当前块的数据
            chunk_df = df.iloc[start_idx:end_idx].copy()
            
            # 处理当前块
            chunk_processed = self._preprocess_impression_data_chunk(chunk_df, split)
            
            # 累积处理后的数据
            all_processed_data.extend(chunk_processed)
            
            # 每处理几个块就保存一次中间结果并清理内存
            if (chunk_idx + 1) % 5 == 0 or chunk_idx == total_chunks - 1:
                logger.info(f"保存中间结果，当前累积{len(all_processed_data)}条记录")
                
                # 转换为DataFrame并保存
                if all_processed_data:
                    temp_df = pd.DataFrame(all_processed_data)
                    
                    # 如果是第一次保存，直接保存；否则追加
                    if not output_path.exists():
                        temp_df.to_pickle(output_path)
                    else:
                        # 读取已有数据，合并后保存
                        existing_df = pd.read_pickle(output_path)
                        combined_df = pd.concat([existing_df, temp_df], ignore_index=True)
                        combined_df.to_pickle(output_path)
                        del existing_df
                    
                    del temp_df
                    all_processed_data = []  # 清空列表释放内存
                
                # 强制垃圾回收
                import gc
                gc.collect()
            
            # 清理当前块数据
            del chunk_df, chunk_processed
            
        logger.info(f"{split}数据分块处理完成")
    
    def _preprocess_impression_data_chunk(self, df: pd.DataFrame, split: str) -> List[Dict]:
        """预处理单个数据块"""
        processed_data = []
        
        # 添加数据采样以减少内存使用
        sample_ratio = self.config.get('data_sample_ratio', 1.0)
        if sample_ratio < 1.0:
            original_len = len(df)
            df = df.sample(frac=sample_ratio, random_state=42)
            logger.info(f"数据采样: 从 {original_len} 条记录采样到 {len(df)} 条 (比例: {sample_ratio})")
        
        processed_count = 0
        max_samples = self.config.get('max_samples_per_split')
        
        for idx, row in df.iterrows():
            if max_samples is not None and processed_count >= max_samples:
                logger.info(f"达到最大样本数限制 {max_samples}，停止处理")
                break
                
            user_id = str(row.get('UserID', ''))
            
            # 构建用户历史 - 从点击历史中提取
            user_history = self._build_user_history_from_row(row)
            
            # 处理正面新闻（点击的）
            pos_news_str = str(row.get('pos', ''))
            if pos_news_str and pos_news_str != 'nan':
                pos_news_ids = pos_news_str.split()
                
                # 限制每个用户的正面样本数量
                max_pos_per_user = self.config.get('max_pos_samples_per_user', 10)
                pos_news_ids = pos_news_ids[:max_pos_per_user]
                
                for news_id in pos_news_ids:
                    if news_id in self.news_corpus:
                        news_info = self.news_corpus[news_id]
                        
                        # 跳过无效数据
                        if not news_info['title'] or len(news_info['title'].split()) < 3:
                            continue
                        
                        processed_row = {
                            'user_id': user_id,
                            'news_id': news_id,
                            'title': news_info['title'],
                            'body': news_info['body'][:self.config.get('max_body_length', 500)],
                            'category': news_info['category'],
                            'user_history': json.dumps(user_history[:self.config.get('max_user_history', 50)]) if user_history else "[]",
                            'split': split,
                            'label': 1  # 正面样本
                        }
                        
                        processed_data.append(processed_row)
                        processed_count += 1
                        
                        if max_samples is not None and processed_count >= max_samples:
                            break
            
            if max_samples is not None and processed_count >= max_samples:
                break
            
            # 处理负面新闻（未点击的）
            neg_news_str = str(row.get('neg', ''))
            if neg_news_str and neg_news_str != 'nan' and self.config.get('include_negative_samples', True):
                neg_news_ids = neg_news_str.split()
                
                # 随机采样负面样本
                neg_sample_ratio = self.config.get('negative_sample_ratio', 0.1)
                max_neg_samples = max(1, int(len(neg_news_ids) * neg_sample_ratio))
                max_neg_per_user = self.config.get('max_neg_samples_per_user', 5)
                max_neg_samples = min(max_neg_samples, max_neg_per_user)
                
                import random
                neg_news_ids = random.sample(neg_news_ids, min(len(neg_news_ids), max_neg_samples))
                
                for news_id in neg_news_ids:
                    if news_id in self.news_corpus:
                        news_info = self.news_corpus[news_id]
                        
                        if not news_info['title'] or len(news_info['title'].split()) < 3:
                            continue
                        
                        processed_row = {
                            'user_id': user_id,
                            'news_id': news_id,
                            'title': news_info['title'],
                            'body': news_info['body'][:self.config.get('max_body_length', 500)],
                            'category': news_info['category'],
                            'user_history': json.dumps(user_history[:self.config.get('max_user_history', 50)]) if user_history else "[]",
                            'split': split,
                            'label': 0  # 负面样本
                        }
                        
                        processed_data.append(processed_row)
                        processed_count += 1
                        
                        if max_samples is not None and processed_count >= max_samples:
                            break
            
            if max_samples is not None and processed_count >= max_samples:
                break
        
        return processed_data
    
    def load_processed_data(self, file_path: str) -> pd.DataFrame:
        """加载已处理的数据"""

        logger.info(f"加载处理后的数据: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"处理后的数据文件不存在: {file_path}")
        
        print(f"1111加载处理后的数据: {file_path}")

        df = pd.read_pickle(file_path)
        print(f"加载完成，包含 {len(df)} 条记录")
        
        # 构建词汇表（如果还未构建）
        if not hasattr(self, 'vocab'):
            self._build_vocabulary(df)
        
        return df
    
    def _build_vocabulary(self, df: pd.DataFrame):
        """构建词汇表 - 优化版本"""
        from collections import Counter
        import json
        import time
        
        # 检查是否已有缓存的词汇表
        vocab_cache_path = 'data/processed/vocab_cache.json'
        if os.path.exists(vocab_cache_path):
            logger.info(f"发现缓存的词汇表: {vocab_cache_path}")
            try:
                with open(vocab_cache_path, 'r', encoding='utf-8') as f:
                    self.vocab = json.load(f)
                self.idx2word = {idx: word for word, idx in self.vocab.items()}
                logger.info(f"从缓存加载词汇表完成，包含 {len(self.vocab)} 个词语")
                return
            except Exception as e:
                logger.warning(f"加载缓存词汇表失败: {str(e)}，将重新构建")
        
        logger.info("开始构建词汇表...")
        start_time = time.time()
        
        # 优化策略1：数据采样 - 只用20%的数据构建词汇表
        vocab_sample_ratio = 0.2
        if len(df) > 10000:  # 只有数据量大时才采样
            sample_size = int(len(df) * vocab_sample_ratio)
            df_sample = df.sample(n=sample_size, random_state=42)
            logger.info(f"使用数据采样构建词汇表: {len(df_sample)}/{len(df)} 条记录 (采样率: {vocab_sample_ratio})")
        else:
            df_sample = df
            logger.info(f"数据量较小，使用全量数据构建词汇表: {len(df)} 条记录")
        
        word_counts = Counter()
        
        # 优化策略2：限制处理的文本长度
        max_title_words = 30  # 标题最多处理30个词
        max_body_words = 100  # 正文最多处理100个词
        max_history_items = 5  # 用户历史最多处理5个新闻
        
        # 统计所有文本中的词语 - 带进度显示
        total_rows = len(df_sample)
        processed_rows = 0
        report_interval = max(1000, total_rows // 10)  # 每处理10%或1000条显示一次进度
        
        for idx, row in df_sample.iterrows():
            # 进度显示
            processed_rows += 1
            if processed_rows % report_interval == 0 or processed_rows == total_rows:
                progress = (processed_rows / total_rows) * 100
                elapsed = time.time() - start_time
                logger.info(f"词汇表构建进度: {processed_rows}/{total_rows} ({progress:.1f}%) - 耗时: {elapsed:.1f}秒")
            
            # 处理标题
            if 'title' in row and pd.notna(row['title']):
                words = str(row['title']).lower().split()[:max_title_words]
                word_counts.update(words)
            
            # 处理正文（限制长度）
            if 'body' in row and pd.notna(row['body']):
                words = str(row['body']).lower().split()[:max_body_words]
                word_counts.update(words)
            
            # 处理用户历史（限制数量和长度）
            if 'user_history' in row and pd.notna(row['user_history']):
                try:
                    history = json.loads(row['user_history'])
                    # 只处理前几个历史记录
                    for item in history[:max_history_items]:
                        if 'title' in item:
                            words = str(item['title']).lower().split()[:max_title_words]
                            word_counts.update(words)
                        if 'body' in item:
                            words = str(item['body']).lower().split()[:max_body_words]
                            word_counts.update(words)
                except:
                    continue
        
        # 构建词汇表
        vocab_size = self.config.get('vocab_size', 5000)  # 降低默认词汇表大小
        min_freq = self.config.get('min_word_freq', 3)    # 提高最低频率要求
        
        # 特殊token
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        # 添加高频词
        logger.info(f"从 {len(word_counts)} 个唯一词语中选择前 {vocab_size-4} 个高频词...")
        for word, count in word_counts.most_common(vocab_size - 4):
            if count >= min_freq:
                self.vocab[word] = len(self.vocab)
        
        # 创建反向词汇表
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        
        # 缓存词汇表
        try:
            os.makedirs(os.path.dirname(vocab_cache_path), exist_ok=True)
            with open(vocab_cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            logger.info(f"词汇表已缓存到: {vocab_cache_path}")
        except Exception as e:
            logger.warning(f"缓存词汇表失败: {str(e)}")
        
        end_time = time.time()
        logger.info(f"词汇表构建完成!")
        logger.info(f"  - 词汇表大小: {len(self.vocab)}")
        logger.info(f"  - 总耗时: {end_time - start_time:.1f}秒")
        logger.info(f"  - 最低词频: {min_freq}")
        logger.info(f"  - 数据采样率: {vocab_sample_ratio if len(df) > 10000 else 1.0}")

def create_preprocessor(config_path: str) -> PENSPreprocessor:
    """创建预处理器"""
    # 加载配置
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                full_config = yaml.safe_load(f)
                # 提取预处理配置
                config = full_config.get('preprocessing', {})
                # 合并其他相关配置
                if 'data_loader' in full_config:
                    config.update(full_config['data_loader'])
            else:
                config = json.load(f)
    else:
        # 默认配置
        config = {
            'data_sample_ratio': 0.01,
            'max_samples_per_split': 5000,
            'max_pos_samples_per_user': 2,
            'max_neg_samples_per_user': 1,
            'negative_sample_ratio': 0.005,
            'max_body_length': 200,
            'max_user_history': 3,
            'include_negative_samples': True,
            'max_title_length': 30,
            'vocab_size': 50000,
            'min_word_freq': 2
        }
    
    return PENSPreprocessor(config)