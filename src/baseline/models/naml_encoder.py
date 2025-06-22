"""
NAML用户编码器
专门用于PENS个性化新闻标题生成的最佳用户建模方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AttentivePooling(nn.Module):
    """注意力池化层"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_layer = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            inputs: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len]
        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # 计算注意力权重
        attention_weights = self.attention_layer(inputs).squeeze(-1)  # [batch_size, seq_len]
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
        
        attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, seq_len]
        
        # 加权池化
        pooled = torch.sum(inputs * attention_weights.unsqueeze(-1), dim=1)  # [batch_size, hidden_dim]
        
        return pooled


class NewsEncoder(nn.Module):
    """新闻编码器 - 多视图学习"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_filters: int = 100, num_categories: int = 15):
        super().__init__()
        
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 标题编码器
        self.title_cnn = nn.Conv1d(embedding_dim, num_filters, kernel_size=3, padding=1)
        self.title_attention = AttentivePooling(num_filters)
        
        # 分类信息编码器
        self.category_embedding = nn.Embedding(num_categories + 1, 100, padding_idx=0)  # +1 for unknown category
        
        # 融合层
        self.fusion_layer = nn.Linear(num_filters + 100, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.word_embedding.weight)
        self.word_embedding.weight.data[0].fill_(0)  # padding token
        
        nn.init.xavier_uniform_(self.category_embedding.weight)
        self.category_embedding.weight.data[0].fill_(0)  # padding token
    
    def forward(self, titles: torch.Tensor, categories: Optional[torch.Tensor] = None, 
                title_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            titles: [batch_size, max_title_length]
            categories: [batch_size] 类别ID
            title_mask: [batch_size, max_title_length]
        """
        batch_size = titles.size(0)
        
        # 标题编码
        title_emb = self.word_embedding(titles)  # [batch_size, max_title_length, embedding_dim]
        title_emb = title_emb.transpose(1, 2)  # [batch_size, embedding_dim, max_title_length]
        title_features = F.relu(self.title_cnn(title_emb))  # [batch_size, num_filters, max_title_length]
        title_features = title_features.transpose(1, 2)  # [batch_size, max_title_length, num_filters]
        
        # 注意力池化
        title_repr = self.title_attention(title_features, title_mask)  # [batch_size, num_filters]
        
        # 类别编码
        if categories is not None:
            category_repr = self.category_embedding(categories)  # [batch_size, 100]
        else:
            category_repr = torch.zeros(batch_size, 100, device=titles.device)
        
        # 特征融合
        combined = torch.cat([title_repr, category_repr], dim=-1)  # [batch_size, num_filters + 100]
        news_repr = self.dropout(F.relu(self.fusion_layer(combined)))  # [batch_size, hidden_dim]
        
        return news_repr


class UserEncoder(nn.Module):
    """用户编码器 - 基于注意力的历史聚合"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.user_attention = AttentivePooling(hidden_dim)
    
    def forward(self, user_history_reprs: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            user_history_reprs: [batch_size, max_history_length, hidden_dim]
            history_mask: [batch_size, max_history_length]
        """
        user_repr = self.user_attention(user_history_reprs, history_mask)
        return user_repr


class NAMLEncoder(nn.Module):
    """
    NAML用户编码器
    论文中表现最佳的用户建模方法
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        max_history_length: int = 50,
        num_filters: int = 100,
        num_categories: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_history_length = max_history_length
        
        # 新闻编码器
        self.news_encoder = NewsEncoder(vocab_size, embedding_dim, hidden_dim, num_filters, num_categories)
        
        # 用户编码器
        self.user_encoder = UserEncoder(hidden_dim)
        
        # 点击预测器（用于预训练）
        self.click_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_news(self, news_titles: torch.Tensor, categories: Optional[torch.Tensor] = None,
                   title_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码单个新闻"""
        return self.news_encoder(news_titles, categories, title_mask)
    
    def forward(self, user_history: torch.Tensor, history_mask: Optional[torch.Tensor] = None,
                history_categories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码用户历史获得用户表示
        
        Args:
            user_history: [batch_size, max_history_length, max_title_length]
            history_mask: [batch_size, max_history_length]
            history_categories: [batch_size, max_history_length]
        
        Returns:
            user_repr: [batch_size, hidden_dim]
        """
        batch_size, max_history, max_title = user_history.size()
        
        # 重塑用户历史用于编码新闻
        history_flat = user_history.view(-1, max_title)  # [batch_size * max_history, max_title]
        
        if history_categories is not None:
            categories_flat = history_categories.view(-1)  # [batch_size * max_history]
        else:
            categories_flat = None
        
        # 创建标题掩码
        title_mask = (history_flat != 0)  # [batch_size * max_history, max_title]
        
        # 编码历史新闻
        history_reprs = self.encode_news(history_flat, categories_flat, title_mask)
        # [batch_size * max_history, hidden_dim]
        
        history_reprs = history_reprs.view(batch_size, max_history, -1)
        # [batch_size, max_history, hidden_dim]
        
        # 用户编码
        user_repr = self.user_encoder(history_reprs, history_mask)
        
        return user_repr
    
    def predict_click(self, user_repr: torch.Tensor, news_repr: torch.Tensor) -> torch.Tensor:
        """预测点击概率（用于预训练）"""
        combined = torch.cat([user_repr, news_repr], dim=-1)
        click_prob = torch.sigmoid(self.click_predictor(combined))
        return click_prob.squeeze(-1)
    
    def load_pretrained_embeddings(self, embeddings: torch.Tensor):
        """加载预训练词嵌入"""
        if embeddings.size() != self.news_encoder.word_embedding.weight.size():
            logger.warning(f"预训练嵌入维度不匹配: {embeddings.size()} vs {self.news_encoder.word_embedding.weight.size()}")
            return
        
        self.news_encoder.word_embedding.weight.data.copy_(embeddings)
        logger.info("已加载预训练词嵌入")
    
    def freeze_embeddings(self):
        """冻结词嵌入层"""
        self.news_encoder.word_embedding.weight.requires_grad = False
        logger.info("已冻结词嵌入层")
    
    def unfreeze_embeddings(self):
        """解冻词嵌入层"""
        self.news_encoder.word_embedding.weight.requires_grad = True
        logger.info("已解冻词嵌入层")
    
    def get_user_embedding_dim(self) -> int:
        """获取用户嵌入维度"""
        return self.hidden_dim