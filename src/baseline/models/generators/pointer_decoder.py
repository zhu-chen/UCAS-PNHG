"""
指针生成网络解码器 - 与原作者实现完全一致
支持F1、F2、F3三种个性化注入策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None
    
    def set_mask(self, mask):
        """设置注意力掩码"""
        self.mask = mask
    
    def forward(self, output, context, user_embed=None):
        batch_size = output.size(0)
        hidden_size = output.size(2)  # 2*dim
        input_size = context.size(1)
        
        attn = torch.bmm(output, context.transpose(1, 2))
        
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        mix = torch.bmm(attn, context)
        
        combined = torch.cat((mix, output), dim=2)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        
        return output, attn


class Attention2(nn.Module):
    """F2个性化注意力机制"""
    
    def __init__(self, dim: int, mask=None):
        super().__init__()
        self.dim = dim
        self.linear_out = nn.Linear(dim * 3, dim)
        self.mask = mask
    
    def set_mask(self, mask):
        """设置注意力掩码"""
        self.mask = mask
    
    def forward(self, output, context, user_embed):
        # context: (bz, 500, 128)
        # output: (bz, seq_len, 128)  
        # user_embed: (bz, 128)
        
        batch_size = output.size(0)
        seq_len = output.size(1)
        hidden_size = output.size(2)
        input_size = context.size(1)
        
        # 标准注意力
        attn1 = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn1.data.masked_fill_(self.mask, -float('inf'))
        attn1 = F.softmax(attn1.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        mix1 = torch.bmm(attn1, context)  # (b, seq_len, 128)
        
        # 个性化注意力
        user_embed_expanded = user_embed.unsqueeze(1).expand(batch_size, seq_len, self.dim)
        attn2 = torch.bmm(user_embed_expanded, context.transpose(1, 2))
        if self.mask is not None:
            attn2.data.masked_fill_(self.mask, -float('inf'))
        attn2 = F.softmax(attn2.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        mix2 = torch.bmm(attn2, context)  # (b, seq_len, 128)
        
        # 融合三个部分：标准注意力、个性化注意力、解码器输出
        combined = torch.cat((mix1, mix2, output), dim=2)
        output = F.tanh(self.linear_out(combined))
        
        return output, attn1


class PointerDecoder(nn.Module):
    """
    指针生成网络解码器
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 400,  # 与Transformer输出匹配
        num_layers: int = 2,
        dropout: float = 0.2,
        max_decode_length: int = 30,
        decoder_type: int = 2,  # 1: F1, 2: F2, 3: F3
        user_size: int = 64  # NAML输出维度
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_decode_length = max_decode_length
        self.decoder_type = decoder_type
        self.user_size = user_size
        
        # 词嵌入层
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM解码器
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        if decoder_type == 1:  # F1: 标准注意力(弃用)
            self.attention = Attention(hidden_dim)
            # F1需要用户嵌入初始化隐状态的变换
            self.transform = nn.ModuleList([
                nn.Linear(user_size, hidden_dim, bias=True) 
                for _ in range(2)  # LSTM有h和c两个状态
            ])
        elif decoder_type == 2:  # F2: 个性化注意力
            self.attention = Attention2(hidden_dim)
            self.transform = nn.Linear(user_size, hidden_dim)
        elif decoder_type == 3:  # F3: 个性化生成概率(弃用)
            self.attention = Attention(hidden_dim)
            self.transform = nn.Linear(user_size, hidden_dim)
            self.transform2output = nn.Linear(2 * hidden_dim, hidden_dim)
        
        # 输出层
        self.out = nn.Linear(hidden_dim, vocab_size)
        
        # 指针生成网络
        self.p_gen_linear = nn.Linear(hidden_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 特殊token ID
        self.sos_id = 2
        self.eos_id = 3
        self.pad_id = 0
    
    def forward_step(self, tgt_input, hidden, encoder_outputs, user_embedding, src=None):
        """
        单步前向传播 
        """
        bz, seq_len = tgt_input.size(0), tgt_input.size(1)
        embedded = self.embeddings(tgt_input)
        embedded = self.dropout(embedded)  # B*S*hidden_dim
        
        output, hidden = self.rnn(embedded, hidden)
        user_temp = None
        
        # 根据decoder_type处理个性化注入
        if self.decoder_type == 1:  # F1: 初始化隐状态
            output_expand = output
        elif self.decoder_type == 2:  # F2: 个性化注意力
            user_temp = self.transform(user_embedding)
            output_expand = output
        elif self.decoder_type == 3:  # F3: 个性化输出
            user_embs = self.transform(user_embedding)
            user_embs = user_embs.unsqueeze(1).expand(output.shape[0], output.shape[1], self.hidden_dim)
            output_expand = self.transform2output(torch.cat((output, user_embs), dim=-1))
        
        # 注意力机制
        if hasattr(self, 'attention'):
            output, attn = self.attention(output_expand, encoder_outputs, user_temp)
        
        # 生成词汇表分布
        predicted_softmax = F.softmax(self.out(output), dim=2)
        
        # 指针生成机制
        p_gen = torch.sigmoid(self.p_gen_linear(output))  # (B, S, 1)
        vocab_dist = p_gen * predicted_softmax
        
        # 扩展源序列用于复制
        doc_len = src.size()[1]
        src_extend = src.unsqueeze(1).expand(bz, seq_len, doc_len)  # (B, S, doc_len)
        attn_ = (1 - p_gen) * attn
        
        # 计算最终分布
        logit = torch.log(vocab_dist.scatter_add(2, src_extend, attn_))
        
        return logit, hidden, attn, output
    
    def _init_state(self, encoder_final):
        """初始化解码器状态"""
        def _fix_enc_hidden(hidden):
            # 处理双向编码器的隐状态
            if isinstance(hidden, tuple):
                hidden = tuple([self._cat_directions(h) for h in hidden])
            else:
                hidden = self._cat_directions(hidden)
            return hidden
        
        if encoder_final is not None:
            return _fix_enc_hidden(encoder_final)
        else:
            # 如果没有编码器最终状态，使用零初始化
            batch_size = 1  # 推理时的批大小
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
            return (h, c)
    
    def _cat_directions(self, hidden):
        """合并双向隐状态"""
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        
        if isinstance(hidden, tuple):
            # LSTM state
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU state
            hidden = _cat(hidden)
        return hidden
    
    def init_hidden_with_user(self, batch_size, device, user_embedding=None):
        """F1策略：使用用户嵌入初始化隐状态"""
        if user_embedding is not None and self.decoder_type == 1:
            # 使用用户嵌入初始化
            h = self.transform[0](user_embedding).unsqueeze(0).repeat(self.num_layers, 1, 1)
            c = self.transform[1](user_embedding).unsqueeze(0).repeat(self.num_layers, 1, 1)
            return (h, c)
        else:
            # 零初始化
            h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            return (h, c)
    
    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        user_embedding: Optional[torch.Tensor] = None,
        injection_method: str = "f2",  # "f1", "f2", "f3"
        teacher_forcing_ratio: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播
        
        Args:
            encoder_outputs: [batch_size, src_len, hidden_dim] 编码器输出
            encoder_input_ids: [batch_size, src_len] 编码器输入ID  
            target_ids: [batch_size, tgt_len] 目标序列
            encoder_mask: [batch_size, src_len] 编码器掩码
            user_embedding: [batch_size, user_dim] 用户嵌入
            injection_method: 个性化注入方式
            teacher_forcing_ratio: 教师强制比例
        """
        batch_size, src_len, _ = encoder_outputs.size()
        device = encoder_outputs.device
        
        # 设置decoder_type
        if injection_method == "f1":
            self.decoder_type = 1
        elif injection_method == "f2":
            self.decoder_type = 2
        elif injection_method == "f3":
            self.decoder_type = 3
        
        # 初始化隐状态
        hidden = self.init_hidden_with_user(batch_size, device, user_embedding)
        
        outputs = {
            "logits": [],
            "attention_weights": [],
            "final_dists": []
        }
        
        # 准备输入
        if target_ids is not None:
            tgt_len = target_ids.size(1)
            # 使用教师强制
            for t in range(tgt_len - 1):
                input_token = target_ids[:, t:t+1]
                
                logit, hidden, attn, _ = self.forward_step(
                    input_token, hidden, encoder_outputs, user_embedding, encoder_input_ids
                )
                
                outputs["logits"].append(logit.squeeze(1))
                outputs["attention_weights"].append(attn.squeeze(1))
                outputs["final_dists"].append(F.softmax(logit.squeeze(1), dim=-1))
        else:
            # 推理模式
            input_token = torch.full((batch_size, 1), self.sos_id, device=device)
            
            for t in range(self.max_decode_length):
                logit, hidden, attn, _ = self.forward_step(
                    input_token, hidden, encoder_outputs, user_embedding, encoder_input_ids
                )
                
                outputs["logits"].append(logit.squeeze(1))
                outputs["attention_weights"].append(attn.squeeze(1))
                final_dist = F.softmax(logit.squeeze(1), dim=-1)
                outputs["final_dists"].append(final_dist)
                
                # 下一个输入
                input_token = torch.argmax(final_dist, dim=-1, keepdim=True)
                
                # 检查EOS
                if torch.all(input_token.squeeze() == self.eos_id):
                    break
        
        # 堆叠输出
        for key in outputs:
            if outputs[key]:
                outputs[key] = torch.stack(outputs[key], dim=1)
        
        return outputs