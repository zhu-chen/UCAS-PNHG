"""
个性化新闻标题生成器 - 与原作者实现完全一致
整合NAML用户编码器 + Transformer编码器 + 指针网络解码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
import logging

from .naml_encoder import NAMLEncoder
from .generators.transformer_encoder import TransformerEncoder
from .generators.pointer_decoder import PointerDecoder

logger = logging.getLogger(__name__)


class PersonalizedHeadlineGenerator(nn.Module):
    """
    个性化新闻标题生成器 - PENS最佳组合
    NAML + F2 + Transformer + Pointer-Generator
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
                "hidden_dim": 400,
                "max_history_length": 50,
                "num_categories": 15,
                "dropout": 0.2
            }
        
        if transformer_config is None:
            transformer_config = {
                "embedding_dim": 300,
                "d_model": 300,
                "sentence_pos_dim": 100,
                "dropout": 0.2,
                "max_length": 500
            }
        
        if decoder_config is None:
            decoder_config = {
                "embedding_dim": 300,
                "hidden_dim": 400,  # 匹配Transformer输出：300+100=400
                "num_layers": 2,
                "dropout": 0.2,
                "max_decode_length": 30,
                "decoder_type": 2,  # F2策略
                "user_size": 64  # NAML输出维度
            }
        
        # 初始化NAML用户编码器
        self.user_encoder = NAMLEncoder(vocab_size=vocab_size, **user_encoder_config)
        
        # 初始化Transformer编码器
        self.transformer_encoder = TransformerEncoder(vocab_size=vocab_size, **transformer_config)
        
        # 初始化指针网络解码器（支持F2注入）
        self.decoder = PointerDecoder(vocab_size=vocab_size, **decoder_config)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        logger.info(f"初始化个性化标题生成器，词汇表大小: {vocab_size}")
    
    def forward(
        self,
        user_history: torch.Tensor,
        news_input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        history_mask: Optional[torch.Tensor] = None,
        history_categories: Optional[torch.Tensor] = None,
        news_attention_mask: Optional[torch.Tensor] = None,
        news_sentence_positions: Optional[torch.Tensor] = None,
        injection_method: str = "f2",
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_history: [batch_size, max_history_length, max_title_length] 用户历史
            news_input_ids: [batch_size, max_body_length] 新闻正文
            target_ids: [batch_size, max_title_length] 目标标题
            history_mask: [batch_size, max_history_length] 历史掩码
            history_categories: [batch_size, max_history_length] 历史类别
            news_attention_mask: [batch_size, max_body_length] 新闻掩码
            news_sentence_positions: [batch_size, max_body_length] 句子位置
            injection_method: 个性化注入方式 ("f1", "f2", "f3")
            teacher_forcing_ratio: 教师强制比例
        """
        
        # 1. 用户编码
        user_embedding = self.user_encoder.encode_user_history(
            user_history, history_mask, history_categories
        )  # [batch_size, 64]
        
        # 2. 新闻编码
        encoder_outputs = self.transformer_encoder(
            input_ids=news_input_ids,
            attention_mask=news_attention_mask,
            sentence_positions=news_sentence_positions
        )  # [batch_size, max_body_length, 400]
        
        # 3. 个性化解码
        decoder_outputs = self.decoder(
            encoder_outputs=encoder_outputs,
            encoder_input_ids=news_input_ids,
            target_ids=target_ids,
            encoder_mask=news_attention_mask,
            user_embedding=user_embedding,
            injection_method=injection_method,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        
        # 4. 计算损失（如果有目标序列）
        loss = None
        if target_ids is not None and "logits" in decoder_outputs:
            # 目标序列去掉SOS token，logits对应预测下一个token
            target_shifted = target_ids[:, 1:]  # 去掉SOS
            logits = decoder_outputs["logits"]
            
            # 确保维度匹配
            min_len = min(target_shifted.size(1), logits.size(1))
            target_shifted = target_shifted[:, :min_len]
            logits = logits[:, :min_len, :]
            
            # 计算损失
            loss = self.criterion(
                logits.contiguous().view(-1, self.vocab_size),
                target_shifted.contiguous().view(-1)
            )
        
        return {
            "loss": loss,
            "user_embedding": user_embedding,
            "encoder_outputs": encoder_outputs,
            **decoder_outputs
        }
    
    def generate(
        self,
        user_history: torch.Tensor,
        news_input_ids: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
        history_categories: Optional[torch.Tensor] = None,
        news_attention_mask: Optional[torch.Tensor] = None,
        news_sentence_positions: Optional[torch.Tensor] = None,
        injection_method: str = "f2",
        max_length: int = 30,
        num_beams: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        生成个性化标题
        """
        self.eval()
        
        with torch.no_grad():
            # 编码用户和新闻
            user_embedding = self.user_encoder.encode_user_history(
                user_history, history_mask, history_categories
            )
            
            encoder_outputs = self.transformer_encoder(
                input_ids=news_input_ids,
                attention_mask=news_attention_mask,
                sentence_positions=news_sentence_positions
            )
            
            # 生成序列
            if num_beams == 1:
                # 贪心解码
                outputs = self.decoder(
                    encoder_outputs=encoder_outputs,
                    encoder_input_ids=news_input_ids,
                    target_ids=None,
                    encoder_mask=news_attention_mask,
                    user_embedding=user_embedding,
                    injection_method=injection_method,
                    teacher_forcing_ratio=0.0
                )
                
                generated_ids = torch.argmax(outputs["final_dists"], dim=-1)
                
            else:
                # Beam search (简化实现)
                batch_size = user_history.size(0)
                device = user_history.device
                
                # 扩展beam维度
                user_embedding = user_embedding.unsqueeze(1).repeat(1, num_beams, 1).view(-1, user_embedding.size(-1))
                encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, num_beams, 1, 1).view(-1, encoder_outputs.size(1), encoder_outputs.size(2))
                news_input_ids = news_input_ids.unsqueeze(1).repeat(1, num_beams, 1).view(-1, news_input_ids.size(1))
                
                if news_attention_mask is not None:
                    news_attention_mask = news_attention_mask.unsqueeze(1).repeat(1, num_beams, 1).view(-1, news_attention_mask.size(1))
                
                outputs = self.decoder(
                    encoder_outputs=encoder_outputs,
                    encoder_input_ids=news_input_ids,
                    target_ids=None,
                    encoder_mask=news_attention_mask,
                    user_embedding=user_embedding,
                    injection_method=injection_method,
                    teacher_forcing_ratio=0.0
                )
                
                generated_ids = torch.argmax(outputs["final_dists"], dim=-1)
                generated_ids = generated_ids.view(batch_size, num_beams, -1)[:, 0, :]  # 取第一个beam
        
        return {
            "generated_ids": generated_ids,
            "attention_weights": outputs.get("attention_weights"),
            "user_embedding": user_embedding
        }
    
    def load_pretrained_embeddings(self, embeddings: torch.Tensor):
        """加载预训练词嵌入到所有组件"""
        # 加载到用户编码器
        self.user_encoder.load_pretrained_embeddings(embeddings)
        
        # 加载到Transformer编码器
        if embeddings.size() == self.transformer_encoder.embeddings.weight.size():
            self.transformer_encoder.embeddings.weight.data.copy_(embeddings)
            logger.info("已加载预训练词嵌入到Transformer编码器")
        
        # 加载到解码器
        if embeddings.size() == self.decoder.embeddings.weight.size():
            self.decoder.embeddings.weight.data.copy_(embeddings)
            logger.info("已加载预训练词嵌入到解码器")
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型参数统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        user_encoder_params = sum(p.numel() for p in self.user_encoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "user_encoder_params": user_encoder_params,
            "transformer_params": transformer_params,
            "decoder_params": decoder_params
        }


def create_naml_f2_generator(vocab_size: int, config: Dict[str, Any] = None) -> PersonalizedHeadlineGenerator:
    """
    创建NAML+F2个性化新闻标题生成器
    
    Args:
        vocab_size: 词汇表大小
        config: 模型配置
        
    Returns:
        个性化标题生成器实例
    """
    if config is None:
        config = {}
    
    # 提取各组件配置
    user_encoder_config = config.get('user_encoder', {})
    transformer_config = config.get('transformer', {})
    decoder_config = config.get('decoder', {})
    
    # 确保F2策略
    decoder_config['decoder_type'] = 2
    
    model = PersonalizedHeadlineGenerator(
        vocab_size=vocab_size,
        user_encoder_config=user_encoder_config,
        transformer_config=transformer_config,
        decoder_config=decoder_config
    )
    
    return model