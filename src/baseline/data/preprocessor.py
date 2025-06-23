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
        预处理原始PENS数据
        
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
        
        # 预处理训练集
        train_processed = self._preprocess_impression_data(train_df, 'train')
        train_path = output_dir / "train_processed.pkl"
        train_processed.to_pickle(train_path)
        
        # 预处理验证集
        valid_processed = self._preprocess_impression_data(valid_df, 'valid')
        valid_path = output_dir / "valid_processed.pkl"
        valid_processed.to_pickle(valid_path)
        
        # 预处理测试集
        test_processed = self._preprocess_personalized_test_data(test_df)
        test_path = output_dir / "test_processed.pkl"
        test_processed.to_pickle(test_path)
        
        # 保存数据统计信息
        stats = self.analyze_data_statistics([str(train_path), str(valid_path), str(test_path)])
        with open(output_dir / "data_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"预处理完成 - 训练: {len(train_processed)}, 骼证: {len(valid_processed)}, 测试: {len(test_processed)}")
        
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
        sample_ratio = self.config.get('data_sample_ratio', 0.1)  # 默认只使用10%的数据
        if sample_ratio < 1.0:
            original_len = len(df)
            df = df.sample(frac=sample_ratio, random_state=42)
            logger.info(f"数据采样: 从 {original_len} 条记录采样到 {len(df)} 条 (比例: {sample_ratio})")
        
        logger.info(f"预处理{split}数据，处理数据包含 {len(df)} 条记录...")
        
        processed_count = 0
        max_samples = self.config.get('max_samples_per_split', 50000)  # 每个split最多处理5万条
        
        for idx, row in df.iterrows():
            if processed_count >= max_samples:
                logger.info(f"达到最大样本数限制 {max_samples}，停止处理")
                break
                
            if idx % 5000 == 0:
                print(f"处理进度: {processed_count}/{min(len(df), max_samples)}")
                
            user_id = str(row.get('UserID', ''))
            
            # 构建用户历史 - 从点击历史中提取
            user_history = self._build_user_history_from_row(row)
            
            # 处理正面新闻（点击的）
            pos_news_str = str(row.get('pos', ''))
            if pos_news_str and pos_news_str != 'nan':
                pos_news_ids = pos_news_str.split()
                
                # 限制每个用户的正面样本数量
                max_pos_per_user = self.config.get('max_pos_samples_per_user', 5)
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
                            'body': news_info['body'][:self.config.get('max_body_length', 500)],  # 限制body长度
                            'category': news_info['category'],
                            'user_history': json.dumps(user_history[:self.config.get('max_user_history', 10)]) if user_history else "[]",  # 减少历史长度
                            'split': split,
                            'label': 1  # 正面样本
                        }
                        
                        processed_data.append(processed_row)
                        processed_count += 1
                        
                        if processed_count >= max_samples:
                            break
            
            if processed_count >= max_samples:
                break
            
            # 处理负面新闻（未点击的）- 采样更少
            neg_news_str = str(row.get('neg', ''))
            if neg_news_str and neg_news_str != 'nan' and self.config.get('include_negative_samples', True):
                neg_news_ids = neg_news_str.split()
                
                # 随机采样更少的负面样本
                neg_sample_ratio = self.config.get('negative_sample_ratio', 0.02)  # 减少到2%
                max_neg_samples = max(1, int(len(neg_news_ids) * neg_sample_ratio))
                max_neg_per_user = self.config.get('max_neg_samples_per_user', 2)  # 每个用户最多2个负样本
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
                            'user_history': json.dumps(user_history[:self.config.get('max_user_history', 10)]) if user_history else "[]",
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