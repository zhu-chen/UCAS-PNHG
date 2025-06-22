"""
PENS个性化新闻标题生成器
实现最佳方法组合：NAML用户编码器 + F2个性化注入策略
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .naml_encoder import NAMLEncoder
from .generators.transformer_encoder import TransformerEncoder
from .generators.pointer_decoder import PointerDecoder


class PersonalizedHeadlineGenerator(nn.Module):
    """
    PENS个性化新闻标题生成器
    专门实现论文中表现最佳的NAML+F2组合
    """
    
    def __init__(
        self,
        vocab_size: int,
        user_encoder_config: Optional[Dict[str, Any]] = None,
        transformer_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # 默认配置
        if user_encoder_config is None:
            user_encoder_config = {
                "embedding_dim": 300,
                "hidden_dim": 256,
                "max_history_length": 50,
                "num_filters": 100,
                "num_categories": 15,
                "dropout": 0.1
            }
        
        if transformer_config is None:
            transformer_config = {
                "d_model": 512,
                "num_heads": 8,
                "num_layers": 6,
                "d_ff": 2048,
                "max_seq_length": 500,
                "max_sentence_length": 50,
                "d_sentence": 100,
                "dropout": 0.1
            }
        
        if decoder_config is None:
            decoder_config = {
                "embedding_dim": 300,
                "hidden_dim": 512,
                "num_layers": 2,
                "dropout": 0.1,
                "max_decode_length": 30
            }
        
        # 初始化NAML用户编码器
        self.user_encoder = NAMLEncoder(vocab_size=vocab_size, **user_encoder_config)
        
        # 初始化Transformer编码器
        self.transformer_encoder = TransformerEncoder(vocab_size=vocab_size, **transformer_config)
        
        # 初始化指针网络解码器（支持F2注入）
        self.pointer_decoder = PointerDecoder(vocab_size=vocab_size, **decoder_config)
        
        # 用户嵌入到解码器隐层维度的投影层
        user_dim = user_encoder_config["hidden_dim"]
        decoder_dim = decoder_config["hidden_dim"]
        
        if user_dim != decoder_dim:
            self.user_projection = nn.Linear(user_dim, decoder_dim)
        else:
            self.user_projection = nn.Identity()
    
    def encode_user(
        self, 
        user_history: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        history_categories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码用户历史获得用户表示
        
        Args:
            user_history: [batch_size, max_history_length, max_title_length]
            history_mask: [batch_size, max_history_length]
            history_categories: [batch_size, max_history_length]
        
        Returns:
            user_embedding: [batch_size, decoder_hidden_dim]
        """
        user_embedding = self.user_encoder(user_history, history_mask, history_categories)
        user_embedding = self.user_projection(user_embedding)
        return user_embedding
    
    def encode_news_body(
        self,
        input_ids: torch.Tensor,
        sentence_positions: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码新闻正文
        
        Args:
            input_ids: [batch_size, seq_len]
            sentence_positions: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            encoder_outputs: [batch_size, seq_len, d_model]
        """
        return self.transformer_encoder(input_ids, sentence_positions, attention_mask)
    
    def generate_headline(
        self,
        encoder_outputs: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        user_embedding: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        使用F2策略生成个性化标题
        
        Args:
            encoder_outputs: [batch_size, src_len, d_model]
            encoder_input_ids: [batch_size, src_len]
            user_embedding: [batch_size, hidden_dim]
            target_ids: [batch_size, tgt_len] 目标序列（训练时）
            encoder_mask: [batch_size, src_len]
            teacher_forcing_ratio: 教师强制比率
        
        Returns:
            decoder_outputs: 解码器输出字典
        """
        return self.pointer_decoder(
            encoder_outputs=encoder_outputs,
            encoder_input_ids=encoder_input_ids,
            target_ids=target_ids,
            encoder_mask=encoder_mask,
            user_embedding=user_embedding,
            injection_method="f2",  # 固定使用F2策略
            teacher_forcing_ratio=teacher_forcing_ratio
        )
    
    def forward(
        self,
        # 用户历史输入
        user_history: torch.Tensor,
        # 新闻正文输入
        news_input_ids: torch.Tensor,
        news_sentence_positions: torch.Tensor,
        # 可选参数（有默认值）
        history_mask: Optional[torch.Tensor] = None,
        history_categories: Optional[torch.Tensor] = None,
        news_attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播
        
        Args:
            user_history: [batch_size, max_history_length, max_title_length]
            news_input_ids: [batch_size, seq_len]
            news_sentence_positions: [batch_size, seq_len]
            history_mask: [batch_size, max_history_length]
            history_categories: [batch_size, max_history_length]
            news_attention_mask: [batch_size, seq_len]
            target_ids: [batch_size, tgt_len]
            teacher_forcing_ratio: 教师强制比率
        
        Returns:
            outputs: 包含所有输出的字典
        """
        # 1. 编码用户（NAML）
        user_embedding = self.encode_user(user_history, history_mask, history_categories)
        
        # 2. 编码新闻正文（Transformer）
        encoder_outputs = self.encode_news_body(
            news_input_ids, news_sentence_positions, news_attention_mask
        )
        
        # 3. 生成个性化标题（F2策略）
        decoder_outputs = self.generate_headline(
            encoder_outputs=encoder_outputs,
            encoder_input_ids=news_input_ids,
            user_embedding=user_embedding,
            target_ids=target_ids,
            encoder_mask=news_attention_mask,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # 整合所有输出
        outputs = {
            "user_embedding": user_embedding,
            "encoder_outputs": encoder_outputs,
            **decoder_outputs
        }
        
        return outputs
    
    def predict_click(
        self,
        user_history: torch.Tensor,
        news_titles: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        history_categories: Optional[torch.Tensor] = None,
        news_categories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测点击概率（用于NAML预训练）
        
        Args:
            user_history: [batch_size, max_history_length, max_title_length]
            news_titles: [batch_size, max_title_length]
            history_mask: [batch_size, max_history_length]
            history_categories: [batch_size, max_history_length]
            news_categories: [batch_size]
        
        Returns:
            click_probs: [batch_size] 点击概率
        """
        # 编码用户
        user_repr = self.user_encoder(user_history, history_mask, history_categories)
        
        # 编码候选新闻
        title_mask = (news_titles != 0)
        news_repr = self.user_encoder.encode_news(news_titles, news_categories, title_mask)
        
        # 预测点击概率
        click_probs = self.user_encoder.predict_click(user_repr, news_repr)
        
        return click_probs
    
    def load_pretrained_embeddings(self, embeddings: torch.Tensor):
        """加载预训练词嵌入到所有组件"""
        self.user_encoder.load_pretrained_embeddings(embeddings)
        self.transformer_encoder.load_pretrained_embeddings(embeddings)
        
        # 为指针解码器加载嵌入
        if embeddings.size() == self.pointer_decoder.embedding.weight.size():
            self.pointer_decoder.embedding.weight.data.copy_(embeddings)
    
    def freeze_user_encoder(self):
        """冻结用户编码器参数（用于标题生成训练）"""
        for param in self.user_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_user_encoder(self):
        """解冻用户编码器参数"""
        for param in self.user_encoder.parameters():
            param.requires_grad = True


def create_naml_f2_generator(
    vocab_size: int,
    config: Optional[Dict[str, Any]] = None
) -> PersonalizedHeadlineGenerator:
    """
    工厂函数：创建NAML+F2个性化标题生成器
    
    Args:
        vocab_size: 词汇表大小
        config: 模型配置
    
    Returns:
        NAML+F2个性化标题生成器实例
    """
    if config is None:
        config = {}
    
    return PersonalizedHeadlineGenerator(
        vocab_size=vocab_size,
        user_encoder_config=config.get("user_encoder", None),
        transformer_config=config.get("transformer", None),
        decoder_config=config.get("decoder", None)
    )