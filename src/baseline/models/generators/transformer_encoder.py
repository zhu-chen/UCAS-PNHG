"""
Transformer编码器 - 与原作者实现一致
PENS框架中的新闻正文编码器，包含双层位置编码和多头自注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEmbedding(nn.Module):
    
    def __init__(self, d_model: int = 300):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        device = x.device
        
        # 创建位置编码
        pe = torch.zeros(seq_len, d_model, device=device)
        position = torch.arange(0, seq_len, device=device).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() *
                            -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 扩展到batch维度
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        return pe


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int = 300, n_heads: int = 20, d_k: int = 20, d_v: int = 20):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        
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


class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    包含双层位置编码：词级别和句子级别
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        d_model: int = 300,
        sentence_pos_dim: int = 100,
        dropout: float = 0.2,
        max_length: int = 500
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.sentence_pos_dim = sentence_pos_dim
        
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 词级别位置编码
        self.pos_encoder = PositionalEmbedding(embedding_dim)
        
        # 句子级别位置编码
        self.sentence_pos_embedding = nn.Parameter(
            torch.randn(max_length, sentence_pos_dim)  # [L, d_s] L为最大句子数
        )
        
        # 多头注意力
        self.attn_body = MultiHeadAttention(
            d_model=embedding_dim + sentence_pos_dim,  # 300 + 100 = 400
            n_heads=20,
            d_k=20,
            d_v=20
        )
        
        # Dropout层
        self.drop_layer = nn.Dropout(p=dropout)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embeddings.weight)
        self.embeddings.weight.data[0].fill_(0)  # padding token
        nn.init.normal_(self.sentence_pos_embedding, 0, 0.1)
    
    def _create_sentence_positions(self, input_ids: torch.Tensor, max_sentence_length: int = 50) -> torch.Tensor:
        """
        创建句子位置索引
        简化实现：每max_sentence_length个词作为一个句子
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        positions = []
        for i in range(seq_len):
            sentence_idx = min(i // max_sentence_length, self.sentence_pos_embedding.size(0) - 1)
            positions.append(sentence_idx)
        
        positions = torch.tensor(positions, device=device).unsqueeze(0).expand(batch_size, -1)
        return positions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sentence_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            sentence_positions: [batch_size, seq_len] 句子位置索引
        
        Returns:
            encoder_outputs: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len = input_ids.size()
        
        # 词嵌入
        embeddings = self.embeddings(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 词级别位置编码
        pos_encoding = self.pos_encoder(embeddings)  # [batch_size, seq_len, embedding_dim]
        embeddings = embeddings + pos_encoding
        
        # 句子级别位置编码
        if sentence_positions is None:
            sentence_positions = self._create_sentence_positions(input_ids)
        
        # 获取句子位置嵌入
        sentence_pos_emb = self.sentence_pos_embedding[sentence_positions]  # [batch_size, seq_len, sentence_pos_dim]
        
        # 拼接词嵌入和句子位置嵌入
        combined_emb = torch.cat([embeddings, sentence_pos_emb], dim=-1)  # [batch_size, seq_len, embedding_dim + sentence_pos_dim]
        
        # Dropout
        combined_emb = self.drop_layer(combined_emb)
        
        # 多头自注意力
        memory_bank = self.attn_body(combined_emb, combined_emb, combined_emb)
        memory_bank = self.drop_layer(memory_bank)
        
        return memory_bank
