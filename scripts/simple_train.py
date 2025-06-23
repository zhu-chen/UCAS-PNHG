#!/usr/bin/env python3
"""
简化的PENS基线模型训练脚本
"""

import os
import sys
import torch
import pandas as pd
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_simple_logger():
    """设置简单日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/simple_training.log')
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_simple_logger()
    logger.info("开始简化训练流程...")
    
    # 检查GPU
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        logger.info(f"使用GPU: {device}")
    else:
        device = torch.device('cpu')
        logger.info("使用CPU")
    
    try:
        # 直接加载预处理的数据
        logger.info("加载训练数据...")
        train_data = pd.read_pickle('data/processed/train_processed.pkl')
        logger.info(f"训练数据加载完成，包含 {len(train_data)} 条记录")
        
        logger.info("加载验证数据...")
        val_data = pd.read_pickle('data/processed/valid_processed.pkl')
        logger.info(f"验证数据加载完成，包含 {len(val_data)} 条记录")
        
        # 显示数据样本
        logger.info("数据样本预览:")
        logger.info(f"列名: {list(train_data.columns)}")
        logger.info(f"第一条记录:")
        for col in train_data.columns:
            sample_value = str(train_data.iloc[0][col])
            if len(sample_value) > 100:
                sample_value = sample_value[:100] + "..."
            logger.info(f"  {col}: {sample_value}")
        
        # 数据统计
        logger.info("数据统计:")
        logger.info(f"  训练集大小: {len(train_data)}")
        logger.info(f"  验证集大小: {len(val_data)}")
        logger.info(f"  唯一用户数: {train_data['user_id'].nunique()}")
        logger.info(f"  唯一新闻数: {train_data['news_id'].nunique()}")
        logger.info(f"  正负样本分布:")
        logger.info(f"    正样本: {(train_data['label'] == 1).sum()}")
        logger.info(f"    负样本: {(train_data['label'] == 0).sum()}")
        
        logger.info("数据加载成功！开始模型训练...")
        
        # 构建简单的词汇表
        logger.info("构建词汇表...")
        from collections import Counter
        import json
        
        word_counts = Counter()
        
        # 从训练数据构建词汇表
        for idx, row in train_data.iterrows():
            if idx % 10000 == 0:
                logger.info(f"处理词汇表进度: {idx}/{len(train_data)}")
            
            # 处理标题
            if pd.notna(row['title']):
                words = str(row['title']).lower().split()
                word_counts.update(words)
            
            # 处理正文（前50个词）
            if pd.notna(row['body']):
                words = str(row['body']).lower().split()[:50]
                word_counts.update(words)
        
        # 构建词汇表
        vocab_size = 5000
        vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        
        for word, count in word_counts.most_common(vocab_size - 4):
            if count >= 3:  # 最少出现3次
                vocab[word] = len(vocab)
        
        logger.info(f"词汇表构建完成，包含 {len(vocab)} 个词语")
        
        # 创建简单的数据集
        logger.info("创建数据集...")
        
        def text_to_indices(text, vocab, max_len=20):
            """将文本转换为索引序列"""
            words = str(text).lower().split()[:max_len]
            indices = [vocab.get(word, vocab['<UNK>']) for word in words]
            # 填充到固定长度
            while len(indices) < max_len:
                indices.append(vocab['<PAD>'])
            return indices[:max_len]
        
        # 准备训练数据
        logger.info("准备训练数据...")
        train_titles = []
        train_bodies = []
        train_labels = []
        
        for idx, row in train_data.iterrows():
            if idx % 20000 == 0:
                logger.info(f"准备数据进度: {idx}/{len(train_data)}")
            
            title_indices = text_to_indices(row['title'], vocab, 15)
            body_indices = text_to_indices(row['body'], vocab, 30)
            
            train_titles.append(title_indices)
            train_bodies.append(body_indices)
            train_labels.append(int(row['label']))
        
        # 转换为PyTorch张量
        logger.info("转换为PyTorch张量...")
        train_titles = torch.tensor(train_titles, dtype=torch.long)
        train_bodies = torch.tensor(train_bodies, dtype=torch.long)
        train_labels = torch.tensor(train_labels, dtype=torch.float)
        
        logger.info(f"训练数据张量形状:")
        logger.info(f"  标题: {train_titles.shape}")
        logger.info(f"  正文: {train_bodies.shape}")
        logger.info(f"  标签: {train_labels.shape}")
        
        # 创建简单的神经网络模型
        logger.info("创建模型...")
        
        class SimplePersonalizationModel(torch.nn.Module):
            def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.title_lstm = torch.nn.LSTM(embed_dim, hidden_dim//2, batch_first=True, bidirectional=True)
                self.body_lstm = torch.nn.LSTM(embed_dim, hidden_dim//2, batch_first=True, bidirectional=True)
                self.classifier = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * 2, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(hidden_dim, 1),
                    torch.nn.Sigmoid()
                )
                
            def forward(self, title_ids, body_ids):
                # 嵌入
                title_embed = self.embedding(title_ids)
                body_embed = self.embedding(body_ids)
                
                # LSTM编码
                title_out, _ = self.title_lstm(title_embed)
                body_out, _ = self.body_lstm(body_embed)
                
                # 平均池化
                title_repr = title_out.mean(dim=1)
                body_repr = body_out.mean(dim=1)
                
                # 拼接并分类
                combined = torch.cat([title_repr, body_repr], dim=1)
                output = self.classifier(combined)
                
                return output.squeeze()
        
        model = SimplePersonalizationModel(len(vocab)).to(device)
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数数量: {total_params:,}")
        
        # 准备训练
        logger.info("开始训练...")
        
        # 创建数据加载器
        from torch.utils.data import TensorDataset, DataLoader
        
        dataset = TensorDataset(train_titles, train_bodies, train_labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        # 训练循环
        num_epochs = 3
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, (titles, bodies, labels) in enumerate(dataloader):
                titles = titles.to(device)
                bodies = bodies.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(titles, bodies)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 500 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{num_epochs} 完成, 平均损失: {avg_loss:.4f}")
        
        logger.info("训练完成！")
        
        # 保存模型
        torch.save(model.state_dict(), 'results/baseline/simple_model.pth')
        logger.info("模型已保存到 results/baseline/simple_model.pth")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise e

if __name__ == '__main__':
    main()