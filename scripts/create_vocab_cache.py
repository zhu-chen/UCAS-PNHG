#!/usr/bin/env python3
"""
快速创建词汇表缓存
"""

import os
import sys
import pandas as pd
import json
import time
from collections import Counter
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_vocab_cache():
    """创建词汇表缓存"""
    print("开始创建词汇表缓存...")
    
    # 加载训练数据
    print("加载训练数据...")
    train_data = pd.read_pickle('data/processed/train_processed.pkl')
    print(f"训练数据加载完成，包含 {len(train_data)} 条记录")
    
    # 使用采样数据构建词汇表（20%采样，约22,650条记录）
    sample_size = min(20000, len(train_data) // 5)  # 使用20,000条或20%的数据
    df_sample = train_data.sample(n=sample_size, random_state=42)
    print(f"使用 {len(df_sample)} 条记录构建词汇表")
    
    # 统计词语频率
    print("统计词语频率...")
    word_counts = Counter()
    
    for idx, row in df_sample.iterrows():
        if idx % 2000 == 0:
            print(f"处理进度: {idx}/{len(df_sample)}")
        
        # 处理标题
        if pd.notna(row['title']):
            words = str(row['title']).lower().split()[:30]  # 标题最多30词
            word_counts.update(words)
        
        # 处理正文
        if pd.notna(row['body']):
            words = str(row['body']).lower().split()[:100]  # 正文最多100词
            word_counts.update(words)
        
        # 处理用户历史
        if pd.notna(row['user_history']):
            try:
                history = json.loads(row['user_history'])
                for item in history[:5]:  # 最多5个历史记录
                    if 'title' in item:
                        words = str(item['title']).lower().split()[:30]
                        word_counts.update(words)
                    if 'body' in item:
                        words = str(item['body']).lower().split()[:50]  # 历史正文更短
                        word_counts.update(words)
            except:
                continue
    
    # 构建词汇表 - 增加到15000个词汇
    vocab_size = 15000  # 增加词汇表大小
    min_freq = 2        # 降低最小频率要求
    
    print(f"构建词汇表，目标大小: {vocab_size}, 最小频率: {min_freq}")
    
    # 特殊token
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<SOS>': 2,
        '<EOS>': 3
    }
    
    # 添加高频词
    added_words = 0
    for word, count in word_counts.most_common():
        if count >= min_freq and len(vocab) < vocab_size:
            vocab[word] = len(vocab)
            added_words += 1
        if len(vocab) >= vocab_size:
            break
    
    print(f"词汇表构建完成:")
    print(f"  - 总词汇数: {len(vocab)}")
    print(f"  - 添加的词汇数: {added_words}")
    print(f"  - 唯一词汇数: {len(word_counts)}")
    print(f"  - 最小频率: {min_freq}")
    
    # 保存词汇表缓存
    cache_path = 'data/processed/vocab_cache.json'
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"词汇表缓存已保存到: {cache_path}")
    return vocab

if __name__ == '__main__':
    vocab = create_vocab_cache()