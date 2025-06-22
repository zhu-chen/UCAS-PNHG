"""
指针网络解码器
PENS框架中的标题生成解码器，包含指针生成网络的复制/生成机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class Attention(nn.Module):
    """注意力机制"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.w_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.b_attn = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_outputs: [batch_size, src_len, hidden_dim]
            decoder_hidden: [batch_size, hidden_dim]
            encoder_mask: [batch_size, src_len]
        
        Returns:
            context: [batch_size, hidden_dim] 上下文向量
            attention_weights: [batch_size, src_len] 注意力权重
        """
        batch_size, src_len, hidden_dim = encoder_outputs.size()
        
        # 扩展解码器隐状态
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
        # [batch_size, src_len, hidden_dim]
        
        # 计算注意力能量
        energy = torch.tanh(
            self.w_h(encoder_outputs) + 
            self.w_s(decoder_hidden_expanded) + 
            self.b_attn
        )  # [batch_size, src_len, hidden_dim]
        
        attention_scores = self.v(energy).squeeze(-1)  # [batch_size, src_len]
        
        # 应用掩码
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(~encoder_mask, float('-inf'))
        
        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, src_len]
        
        # 计算上下文向量
        context = torch.sum(encoder_outputs * attention_weights.unsqueeze(-1), dim=1)
        # [batch_size, hidden_dim]
        
        return context, attention_weights


class PointerGenerator(nn.Module):
    """指针生成器模块"""
    
    def __init__(self, hidden_dim: int, input_dim: int):
        super().__init__()
        
        # 生成概率计算
        self.w_h = nn.Linear(hidden_dim, 1)      # 解码器隐状态
        self.w_s = nn.Linear(hidden_dim, 1)      # 上下文向量
        self.w_x = nn.Linear(input_dim, 1)       # 解码器输入
        
    def forward(self, context: torch.Tensor, decoder_hidden: torch.Tensor,
                decoder_input: torch.Tensor) -> torch.Tensor:
        """
        计算生成概率 p_gen
        
        Args:
            context: [batch_size, hidden_dim]
            decoder_hidden: [batch_size, hidden_dim]
            decoder_input: [batch_size, input_dim]
        
        Returns:
            p_gen: [batch_size] 生成概率
        """
        p_gen = torch.sigmoid(
            self.w_h(decoder_hidden) + 
            self.w_s(context) + 
            self.w_x(decoder_input)
        ).squeeze(-1)
        
        return p_gen


class PointerDecoder(nn.Module):
    """
    指针网络解码器
    实现PENS框架中的个性化标题生成解码器
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_decode_length: int = 30
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_decode_length = max_decode_length
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=embedding_dim + hidden_dim,  # 输入嵌入 + 上下文向量
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = Attention(hidden_dim)
        
        # 指针生成器
        self.pointer_generator = PointerGenerator(hidden_dim, embedding_dim + hidden_dim)
        
        # 词汇表分布计算
        self.vocab_projection = nn.Linear(hidden_dim + hidden_dim + embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0].fill_(0)  # padding token
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def init_hidden(self, batch_size: int, device: torch.device,
                   user_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化解码器隐状态
        
        Args:
            batch_size: 批大小
            device: 设备
            user_embedding: [batch_size, hidden_dim] 用户嵌入（用于个性化注入F1）
        """
        if user_embedding is not None:
            # F1: 使用用户嵌入初始化解码器隐状态
            h_0 = user_embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)
            c_0 = torch.zeros_like(h_0)
        else:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        
        return h_0, c_0
    
    def forward_step(
        self,
        input_token: torch.Tensor,
        hidden_state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        user_embedding: Optional[torch.Tensor] = None,
        injection_method: str = "none"
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        解码器单步前向传播
        
        Args:
            input_token: [batch_size] 输入token
            hidden_state: LSTM隐状态
            encoder_outputs: [batch_size, src_len, hidden_dim] 编码器输出
            encoder_mask: [batch_size, src_len] 编码器掩码
            user_embedding: [batch_size, hidden_dim] 用户嵌入
            injection_method: 个性化注入方式 ("f1", "f2", "f3", "none")
        
        Returns:
            vocab_dist: [batch_size, vocab_size] 词汇表分布
            hidden_state: 新的隐状态
            attention_weights: [batch_size, src_len] 注意力权重
            p_gen: [batch_size] 生成概率
        """
        batch_size = input_token.size(0)
        
        # 词嵌入
        embedded = self.embedding(input_token)  # [batch_size, embedding_dim]
        
        # 计算注意力和上下文向量
        decoder_hidden = hidden_state[0][-1]  # 使用最后一层的隐状态
        context, attention_weights = self.attention(encoder_outputs, decoder_hidden, encoder_mask)
        
        # F2: 个性化注意力分布
        if injection_method == "f2" and user_embedding is not None:
            # 使用用户嵌入调整注意力权重
            user_attention_bias = torch.matmul(encoder_outputs, user_embedding.unsqueeze(-1)).squeeze(-1)
            # [batch_size, src_len]
            
            if encoder_mask is not None:
                user_attention_bias = user_attention_bias.masked_fill(~encoder_mask, float('-inf'))
            
            user_attention_weights = F.softmax(user_attention_bias, dim=-1)
            
            # 融合原始注意力和用户个性化注意力
            alpha = 0.5  # 融合权重，可作为超参数
            attention_weights = alpha * attention_weights + (1 - alpha) * user_attention_weights
            
            # 重新计算上下文向量
            context = torch.sum(encoder_outputs * attention_weights.unsqueeze(-1), dim=1)
        
        # 拼接嵌入和上下文
        lstm_input = torch.cat([embedded, context], dim=-1)  # [batch_size, embedding_dim + hidden_dim]
        lstm_input = lstm_input.unsqueeze(1)  # [batch_size, 1, input_size]
        
        # LSTM前向传播
        lstm_output, hidden_state = self.lstm(lstm_input, hidden_state)
        lstm_output = lstm_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # 计算词汇表分布
        vocab_input = torch.cat([lstm_output, context, embedded], dim=-1)
        vocab_logits = self.vocab_projection(vocab_input)  # [batch_size, vocab_size]
        vocab_dist = F.softmax(vocab_logits, dim=-1)
        
        # 计算生成概率
        p_gen = self.pointer_generator(context, lstm_output, lstm_input.squeeze(1))
        
        # F3: 个性化生成概率
        if injection_method == "f3" and user_embedding is not None:
            # 使用用户嵌入调整生成概率
            user_gen_bias = torch.matmul(user_embedding, lstm_output.unsqueeze(-1)).squeeze(-1)
            user_gen_bias = torch.sigmoid(user_gen_bias)
            
            # 融合原始生成概率和用户个性化偏好
            beta = 0.5  # 融合权重，可作为超参数
            p_gen = beta * p_gen + (1 - beta) * user_gen_bias
        
        return vocab_dist, hidden_state, attention_weights, p_gen
    
    def compute_final_distribution(
        self,
        vocab_dist: torch.Tensor,
        attention_weights: torch.Tensor,
        p_gen: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        oov_list: Optional[list] = None
    ) -> torch.Tensor:
        """
        计算最终的词汇分布（生成 + 复制）
        
        Args:
            vocab_dist: [batch_size, vocab_size] 词汇表分布
            attention_weights: [batch_size, src_len] 注意力权重
            p_gen: [batch_size] 生成概率
            encoder_input_ids: [batch_size, src_len] 编码器输入token
            oov_list: OOV词汇列表
        
        Returns:
            final_dist: [batch_size, extended_vocab_size] 最终分布
        """
        batch_size, src_len = encoder_input_ids.size()
        
        # 扩展词汇表大小（包含OOV词汇）
        if oov_list is not None:
            extended_vocab_size = self.vocab_size + len(oov_list)
        else:
            extended_vocab_size = self.vocab_size
        
        # 初始化最终分布
        final_dist = torch.zeros(batch_size, extended_vocab_size, device=vocab_dist.device)
        
        # 生成部分：p_gen * P_vocab
        p_gen_expanded = p_gen.unsqueeze(-1)  # [batch_size, 1]
        final_dist[:, :self.vocab_size] = p_gen_expanded * vocab_dist
        
        # 复制部分：(1 - p_gen) * attention_weights
        p_copy = (1 - p_gen).unsqueeze(-1)  # [batch_size, 1]
        copy_dist = p_copy * attention_weights  # [batch_size, src_len]
        
        # 将复制分布加到最终分布中
        final_dist.scatter_add_(1, encoder_input_ids, copy_dist)
        
        return final_dist
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        user_embedding: Optional[torch.Tensor] = None,
        injection_method: str = "none",
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        解码器前向传播
        
        Args:
            encoder_outputs: [batch_size, src_len, hidden_dim]
            encoder_input_ids: [batch_size, src_len]
            target_ids: [batch_size, tgt_len] 目标序列（训练时使用）
            encoder_mask: [batch_size, src_len]
            user_embedding: [batch_size, hidden_dim]
            injection_method: 个性化注入方式
            teacher_forcing_ratio: 教师强制比率
        """
        batch_size, src_len, _ = encoder_outputs.size()
        device = encoder_outputs.device
        
        # 初始化隐状态
        if injection_method == "f1":
            hidden_state = self.init_hidden(batch_size, device, user_embedding)
        else:
            hidden_state = self.init_hidden(batch_size, device)
        
        outputs = {
            "vocab_dists": [],
            "attention_weights": [],
            "p_gens": [],
            "final_dists": []
        }
        
        # 开始token
        input_token = torch.full((batch_size,), 2, device=device)  # 假设SOS token id = 2
        
        for t in range(self.max_decode_length):
            # 单步解码
            vocab_dist, hidden_state, attn_weights, p_gen = self.forward_step(
                input_token, hidden_state, encoder_outputs, encoder_mask,
                user_embedding, injection_method
            )
            
            # 计算最终分布
            final_dist = self.compute_final_distribution(
                vocab_dist, attn_weights, p_gen, encoder_input_ids
            )
            
            # 保存输出
            outputs["vocab_dists"].append(vocab_dist)
            outputs["attention_weights"].append(attn_weights)
            outputs["p_gens"].append(p_gen)
            outputs["final_dists"].append(final_dist)
            
            # 确定下一个输入token
            if target_ids is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # 教师强制
                if t < target_ids.size(1) - 1:
                    input_token = target_ids[:, t + 1]
                else:
                    break
            else:
                # 使用模型预测
                input_token = torch.argmax(final_dist, dim=-1)
        
        # 堆叠输出
        for key in outputs:
            if outputs[key]:
                outputs[key] = torch.stack(outputs[key], dim=1)
        
        return outputs