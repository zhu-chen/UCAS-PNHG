"""
PENS数据集处理器
处理个性化新闻标题生成数据集的加载、预处理和批处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)


class PENSDataset(Dataset):
    """PENS数据集类"""
    
    def __init__(
        self,
        data_source,  # 可以是文件路径或DataFrame
        vocab: Dict[str, int],
        max_title_length: int = 30,
        max_body_length: int = 500,
        max_sentence_length: int = 50,
        max_user_history: int = 50,
        mode: str = "train"
    ):
        """
        Args:
            data_source: 数据文件路径或已加载的DataFrame
            vocab: 词汇表
            max_title_length: 最大标题长度
            max_body_length: 最大正文长度
            max_sentence_length: 最大句子长度
            max_user_history: 最大用户历史长度
            mode: 数据模式 ("train", "valid", "test")
        """
        self.vocab = vocab
        self.max_title_length = max_title_length
        self.max_body_length = max_body_length
        self.max_sentence_length = max_sentence_length
        self.max_user_history = max_user_history
        self.mode = mode
        
        # 判断输入类型
        if isinstance(data_source, (str, Path)):
            self.data_path = Path(data_source)
            self.df = None
        else:
            # 假设是DataFrame
            self.data_path = None
            self.df = data_source
        
        # 特殊token
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        
        self.pad_id = vocab.get(self.pad_token, 0)
        self.unk_id = vocab.get(self.unk_token, 1)
        self.sos_id = vocab.get(self.sos_token, 2)
        self.eos_id = vocab.get(self.eos_token, 3)
        
        # 加载数据
        self.data = self._load_data()
        logger.info(f"加载了 {len(self.data)} 条 {mode} 数据")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """加载数据"""
        if self.df is not None:
            # 直接使用已加载的DataFrame
            df = self.df
        else:
            # 从文件加载
            if self.data_path.suffix == '.pkl':
                with open(self.data_path, 'rb') as f:
                    df = pickle.load(f)
            elif self.data_path.suffix == '.tsv':
                df = pd.read_csv(self.data_path, sep='\t')
            else:
                raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
        
        data = []
        for _, row in df.iterrows():
            try:
                # 解析用户历史
                if 'user_history' in row and pd.notna(row['user_history']):
                    user_history = self._parse_user_history(row['user_history'])
                else:
                    user_history = []
                
                # 解析新闻内容
                title = str(row.get('title', ''))
                body = str(row.get('body', ''))
                category = row.get('category', 'unknown')
                
                # 个性化标题（仅测试集有）
                personalized_title = str(row.get('personalized_title', '')) if 'personalized_title' in row else None
                
                data.append({
                    'user_id': row.get('user_id', ''),
                    'news_id': row.get('news_id', ''),
                    'user_history': user_history,
                    'title': title,
                    'body': body,
                    'category': category,
                    'personalized_title': personalized_title
                })
            except Exception as e:
                logger.warning(f"跳过无效数据行: {e}")
                continue
        
        return data
    
    def _parse_user_history(self, history_str: str) -> List[Dict[str, str]]:
        """解析用户历史"""
        try:
            if isinstance(history_str, str):
                # 假设历史是JSON格式或特定分隔符
                if history_str.startswith('['):
                    history = json.loads(history_str)
                else:
                    # 简单分隔符格式
                    history_items = history_str.split('|')
                    history = [{'title': item, 'category': 'unknown'} for item in history_items]
            else:
                history = []
            
            return history[:self.max_user_history]  # 限制历史长度
        except:
            return []
    
    def _tokenize_text(self, text: str) -> List[int]:
        """文本分词并转换为ID"""
        if not text or pd.isna(text):
            return []
        
        # 简单的空格分词（实际应用中可使用更复杂的分词器）
        tokens = text.lower().strip().split()
        token_ids = [self.vocab.get(token, self.unk_id) for token in tokens]
        
        return token_ids
    
    def _pad_sequence(self, seq: List[int], max_length: int, pad_value: int = None) -> List[int]:
        """填充序列到固定长度"""
        if pad_value is None:
            pad_value = self.pad_id
        
        if len(seq) >= max_length:
            return seq[:max_length]
        else:
            return seq + [pad_value] * (max_length - len(seq))
    
    def _create_sentence_positions(self, body_tokens: List[int]) -> List[int]:
        """创建句子位置索引（简化实现）"""
        # 简化：每50个词作为一个句子
        positions = []
        for i, token in enumerate(body_tokens):
            sentence_idx = min(i // 50, self.max_sentence_length - 1)
            positions.append(sentence_idx)
        
        return positions
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        item = self.data[idx]
        
        # 处理用户历史
        user_history_tokens = []
        history_categories = []
        
        for hist_item in item['user_history']:
            hist_tokens = self._tokenize_text(hist_item.get('title', ''))
            hist_tokens = self._pad_sequence(hist_tokens, self.max_title_length)
            user_history_tokens.append(hist_tokens)
            
            # 类别映射（简化）
            category = hist_item.get('category', 'unknown')
            category_id = hash(category) % 15  # 简单哈希映射到15个类别
            history_categories.append(category_id)
        
        # 填充用户历史到固定长度
        while len(user_history_tokens) < self.max_user_history:
            user_history_tokens.append([self.pad_id] * self.max_title_length)
            history_categories.append(0)  # padding category
        
        user_history_tokens = user_history_tokens[:self.max_user_history]
        history_categories = history_categories[:self.max_user_history]
        
        # 处理新闻正文
        body_tokens = self._tokenize_text(item['body'])
        body_tokens = self._pad_sequence(body_tokens, self.max_body_length)
        sentence_positions = self._create_sentence_positions(body_tokens)
        
        # 处理目标标题
        if self.mode == "test" and item['personalized_title']:
            target_text = item['personalized_title']
        else:
            target_text = item['title']
        
        target_tokens = self._tokenize_text(target_text)
        target_tokens = [self.sos_id] + target_tokens + [self.eos_id]
        target_tokens = self._pad_sequence(target_tokens, self.max_title_length + 2)
        
        # 新闻类别
        news_category = hash(item['category']) % 15
        
        # 创建掩码
        history_mask = [1 if any(tokens) else 0 for tokens in user_history_tokens]
        body_mask = [1 if token != self.pad_id else 0 for token in body_tokens]
        target_mask = [1 if token != self.pad_id else 0 for token in target_tokens]
        
        return {
            'user_history': torch.tensor(user_history_tokens, dtype=torch.long),
            'history_mask': torch.tensor(history_mask, dtype=torch.bool),
            'history_categories': torch.tensor(history_categories, dtype=torch.long),
            'news_input_ids': torch.tensor(body_tokens, dtype=torch.long),
            'news_sentence_positions': torch.tensor(sentence_positions, dtype=torch.long),
            'news_attention_mask': torch.tensor(body_mask, dtype=torch.bool),
            'news_category': torch.tensor(news_category, dtype=torch.long),
            'target_ids': torch.tensor(target_tokens, dtype=torch.long),
            'target_mask': torch.tensor(target_mask, dtype=torch.bool),
            'user_id': item['user_id'],
            'news_id': item['news_id']
        }
    
    def collate_fn(self, batch):
        """批处理函数"""
        # 简单的默认批处理，直接使用PyTorch的默认collate
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)


def build_vocab(data_paths: List[str], vocab_size: int = 50000, min_freq: int = 2) -> Dict[str, int]:
    """构建词汇表"""
    word_count = {}
    
    for data_path in data_paths:
        logger.info(f"处理文件: {data_path}")
        
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                df = pickle.load(f)
        else:
            df = pd.read_csv(data_path, sep='\t')
        
        for _, row in df.iterrows():
            # 处理标题
            if 'title' in row and pd.notna(row['title']):
                tokens = str(row['title']).lower().strip().split()
                for token in tokens:
                    word_count[token] = word_count.get(token, 0) + 1
            
            # 处理正文
            if 'body' in row and pd.notna(row['body']):
                tokens = str(row['body']).lower().strip().split()
                for token in tokens:
                    word_count[token] = word_count.get(token, 0) + 1
            
            # 处理个性化标题
            if 'personalized_title' in row and pd.notna(row['personalized_title']):
                tokens = str(row['personalized_title']).lower().strip().split()
                for token in tokens:
                    word_count[token] = word_count.get(token, 0) + 1
    
    # 过滤低频词
    filtered_words = [word for word, count in word_count.items() if count >= min_freq]
    
    # 按频率排序
    filtered_words.sort(key=lambda x: word_count[x], reverse=True)
    
    # 构建词汇表
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1, 
        "<SOS>": 2,
        "<EOS>": 3
    }
    
    for i, word in enumerate(filtered_words[:vocab_size-4]):
        vocab[word] = i + 4
    
    logger.info(f"构建词汇表完成，总词汇量: {len(vocab)}")
    return vocab


def create_data_loaders(
    train_path: str,
    valid_path: str,
    test_path: str,
    vocab: Dict[str, int],
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 创建数据集
    train_dataset = PENSDataset(train_path, vocab, mode="train", **dataset_kwargs)
    valid_dataset = PENSDataset(valid_path, vocab, mode="valid", **dataset_kwargs)
    test_dataset = PENSDataset(test_path, vocab, mode="test", **dataset_kwargs)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, valid_loader, test_loader


def load_glove_embeddings(glove_path: str, vocab: Dict[str, int], embedding_dim: int = 300) -> torch.Tensor:
    """加载GloVe预训练词嵌入"""
    embeddings = torch.zeros(len(vocab), embedding_dim)
    
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vector = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                embeddings[vocab[word]] = vector
    
    logger.info(f"加载了 {len(vocab)} 个词的预训练嵌入")
    return embeddings