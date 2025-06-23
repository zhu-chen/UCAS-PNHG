"""
Transformer编码器
PENS框架中的新闻正文编码器，包含双层位置编码和多头自注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class SentencePositionalEncoding(nn.Module):
    """句子级位置编码"""
    
    def __init__(self, max_sentences: int, d_sentence: int):
        super().__init__()
        self.sentence_position_embedding = nn.Embedding(max_sentences, d_sentence)
        
    def forward(self, sentence_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence_positions: [batch_size, seq_len] 句子位置索引
        """
        return self.sentence_position_embedding(sentence_positions)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层 - 使用PyTorch内置的MultiheadAttention"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, batch_first: bool = True):
        super().__init__()
        
        # 使用PyTorch内置的多头注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model] if batch_first=True
            key_padding_mask: [batch_size, seq_len] padding mask (True for padding positions)
        """
        # 自注意力 + 残差连接
        # PyTorch的MultiheadAttention期望padding mask中True表示需要被忽略的位置
        attention_output, _ = self.self_attention(
            query=x,
            key=x, 
            value=x,
            key_padding_mask=key_padding_mask,  # True for positions to ignore
            need_weights=False
        )
        x = self.norm1(x + self.dropout(attention_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
    PENS框架的Transformer编码器
    包含双层位置编码和多头自注意力机制
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_length: int = 500,
        max_sentence_length: int = 50,
        d_sentence: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 词嵌入
        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 词级别位置编码
        self.word_pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # 句子级别位置编码
        self.sentence_pos_encoding = SentencePositionalEncoding(max_sentence_length, d_sentence)
        
        # 投影层将词嵌入和句子位置嵌入拼接后的维度投影到d_model
        self.input_projection = nn.Linear(d_model + d_sentence, d_model)
        
        # Transformer编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重 - 改进数值稳定性"""
        # 使用更保守的权重初始化
        nn.init.xavier_uniform_(self.word_embedding.weight, gain=0.1)
        self.word_embedding.weight.data[0].fill_(0)  # padding token
        
        # 为投影层使用更小的初始化
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.1)
        if self.input_projection.bias is not None:
            nn.init.constant_(self.input_projection.bias, 0)
        
        # 为Transformer层使用更保守的初始化
        for layer in self.layers:
            for module in layer.feed_forward.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        sentence_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len] 词ID序列
            sentence_positions: [batch_size, seq_len] 句子位置索引
            attention_mask: [batch_size, seq_len] 注意力掩码 (True for valid positions)
        
        Returns:
            encoded_output: [batch_size, seq_len, d_model] 编码后的表示
        """
        batch_size, seq_len = input_ids.size()
        
        # 词嵌入
        word_embeddings = self.word_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # 词级别位置编码（需要转置以匹配PositionalEncoding的期望输入）
        word_embeddings = word_embeddings.transpose(0, 1)  # [seq_len, batch_size, d_model]
        word_embeddings = self.word_pos_encoding(word_embeddings)
        word_embeddings = word_embeddings.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # 句子级别位置编码
        sentence_embeddings = self.sentence_pos_encoding(sentence_positions)
        # [batch_size, seq_len, d_sentence]
        
        # 拼接词嵌入和句子位置嵌入
        combined_embeddings = torch.cat([word_embeddings, sentence_embeddings], dim=-1)
        # [batch_size, seq_len, d_model + d_sentence]
        
        # 投影到模型维度
        x = self.input_projection(combined_embeddings)  # [batch_size, seq_len, d_model]
        x = self.dropout(x)
        
        # 准备padding mask (PyTorch期望True表示需要被忽略的位置)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # 反转mask，True表示padding位置
        
        # 通过Transformer编码器层
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        
        return x
    
    def load_pretrained_embeddings(self, embeddings: torch.Tensor):
        """加载预训练词嵌入"""
        if embeddings.size() != self.word_embedding.weight.size():
            raise ValueError(f"嵌入维度不匹配: {embeddings.size()} vs {self.word_embedding.weight.size()}")
        
        self.word_embedding.weight.data.copy_(embeddings)
    
    def freeze_embeddings(self):
        """冻结词嵌入层"""
        self.word_embedding.weight.requires_grad = False
    
    def unfreeze_embeddings(self):
        """解冻词嵌入层"""
        self.word_embedding.weight.requires_grad = True