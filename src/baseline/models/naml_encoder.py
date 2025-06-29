"""
NAML用户编码器 - 与原作者实现完全一致
基于PENS论文中的最佳用户建模方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int = 300, n_heads: int = 20, d_k: int = 20, d_v: int = 20):
        super().__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20  
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300 -> 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300 -> 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300 -> 400
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
    
    def forward(self, Q, K, V, attn_mask=None):
        max_len = Q.size(1)
        batch_size = Q.size(0)
        
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, max_len, max_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_s)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        return context


class AttentionPooling(nn.Module):

    def __init__(self, d_h: int, hidden_size: int):
        super().__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=0.2)
    
    def forward(self, x, attn_mask=None):
        # x: [bz, seq_len, d_h]
        bz = x.shape[0]
        e = self.att_fc1(x)  # (bz, seq_len, hidden_size//2)
        e = torch.tanh(e)
        alpha = self.att_fc2(e)  # (bz, seq_len, 1)
        
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, d_h)
        return x


class NAMLEncoder(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 400,  # 原作者使用400
        max_history_length: int = 50,
        num_categories: int = 15,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_history_length = max_history_length
        
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 类别嵌入层 
        self.vert_embed = nn.Embedding(num_categories + 1, 400, padding_idx=0)
        
        # 多头注意力层
        self.attn_title = MultiHeadAttention(300, 20, 20, 20)
        self.attn_body = MultiHeadAttention(300, 20, 20, 20)
        
        # 注意力池化层
        self.title_attn_pool = AttentionPooling(400, 400)
        self.body_attn_pool = AttentionPooling(400, 400)
        self.news_attn_pool = AttentionPooling(400, 400)
        self.attn_pool_news = AttentionPooling(64, 64)
        
        # Dropout层
        self.drop_layer = nn.Dropout(p=dropout)
        
        # 输出层
        self.fc = nn.Linear(400, 64)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def news_encoder(self, news_feature):

        title, vert, body = news_feature[0], news_feature[1], news_feature[2]
        
        # 标题编码
        news_len = title.shape[-1]
        title = title.reshape(-1, news_len)
        title = self.drop_layer(self.embed(title))
        title = self.drop_layer(self.attn_title(title, title, title))
        title = self.title_attn_pool(title).reshape(-1, 1, 400)
        
        # 正文编码
        body_len = body.shape[-1]
        body = body.reshape(-1, body_len)
        body = self.drop_layer(self.embed(body))
        body = self.drop_layer(self.attn_body(body, body, body))
        body = self.body_attn_pool(body).reshape(-1, 1, 400)
        
        # 类别编码
        vert = self.drop_layer(self.vert_embed(vert.reshape(-1))).reshape(-1, 1, 400)
        
        # 多视图融合
        news_vec = torch.cat((title, body, vert), 1)
        news_vec = self.news_attn_pool(news_vec)
        news_vec = self.fc(news_vec)
        
        return news_vec
    
    def user_encoder(self, x):

        x = self.attn_pool_news(x).reshape(-1, 64)
        return x
    
    def forward(self, user_feature, news_feature, label=None, compute_loss=True):

        bz = label.size(0)
        news_vecs = self.news_encoder(news_feature).reshape(bz, -1, 64)
        
        user_newsvecs = self.news_encoder(user_feature).reshape(bz, -1, 64)
        user_vec = self.user_encoder(user_newsvecs).unsqueeze(-1)  # batch * 64 * 1
        score = torch.bmm(news_vecs, user_vec).squeeze(-1)
        
        if compute_loss:
            loss = self.criterion(score, label)
            return loss, score
        else:
            return score
    
    def encode_user_history(self, user_history, history_mask=None, history_categories=None):

        batch_size, max_history, max_title = user_history.size()
        
        # 创建虚拟的body（在NAML中主要用标题）
        user_body = user_history  # 简化处理，实际可以使用更复杂的body信息
        
        # 构造用户特征格式 [title, category, body]
        if history_categories is None:
            history_categories = torch.zeros(batch_size, max_history, dtype=torch.long, device=user_history.device)
        
        user_feature = [
            user_history,  # titles
            history_categories,  # categories  
            user_body  # body (使用标题作为简化)
        ]
        
        # 使用新闻编码器编码用户历史
        user_newsvecs = self.news_encoder(user_feature).reshape(batch_size, -1, 64)
        
        # 使用用户编码器聚合
        user_vec = self.user_encoder(user_newsvecs)
        
        return user_vec
    
    def load_pretrained_embeddings(self, embeddings: torch.Tensor):
        """加载预训练词嵌入"""
        if embeddings.size() != self.embed.weight.size():
            logger.warning(f"预训练嵌入维度不匹配: {embeddings.size()} vs {self.embed.weight.size()}")
            return
        
        self.embed.weight.data.copy_(embeddings)
        logger.info("已加载预训练词嵌入")
    
    def get_user_embedding_dim(self) -> int:
        """获取用户嵌入维度"""
        return 64  
