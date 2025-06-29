#!/usr/bin/env python3
"""
NaN损失诊断脚本 - 精确定位问题所在
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baseline.data.dataset import create_data_loaders
from src.baseline.models.personalized_generator import PersonalizedHeadlineGenerator
from src.baseline.utils.config import load_configs  # 修正函数名

def check_tensor_health(tensor, name="Tensor"):
    """检查tensor的数值健康状况"""
    if tensor is None:
        print(f"❌ {name}: None")
        return False
    
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    min_val = tensor.min().item() if not has_nan else "NaN"
    max_val = tensor.max().item() if not has_nan else "NaN"
    mean_val = tensor.mean().item() if not has_nan else "NaN"
    
    print(f"🔍 {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    print(f"   范围: [{min_val}, {max_val}], 均值: {mean_val}")
    print(f"   NaN: {has_nan.item()}, Inf: {has_inf.item()}")
    
    if has_nan or has_inf:
        print(f"❌ {name} 包含异常值!")
        return False
    
    # 检查梯度范围是否合理
    if hasattr(tensor, 'grad') and tensor.grad is not None:
        grad_has_nan = torch.isnan(tensor.grad).any()
        grad_has_inf = torch.isinf(tensor.grad).any()
        grad_norm = tensor.grad.norm().item() if not grad_has_nan else "NaN"
        print(f"   梯度: norm={grad_norm}, NaN: {grad_has_nan.item()}, Inf: {grad_has_inf.item()}")
        
        if grad_has_nan or grad_has_inf:
            print(f"❌ {name} 梯度包含异常值!")
            return False
    
    print(f"✅ {name} 数值健康")
    return True

def diagnose_model_forward(model, batch, device):
    """诊断模型前向传播过程"""
    print("\n=== 🔍 模型前向传播诊断 ===")
    
    model.eval()
    
    # 检查输入数据
    print("\n1. 检查输入数据:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            check_tensor_health(value, f"输入-{key}")
    
    try:
        # 逐步执行前向传播
        print("\n2. 执行前向传播...")
        
        with torch.no_grad():
            # 用户编码
            print("   -> 用户编码阶段")
            user_history = batch['user_history'].to(device)
            history_mask = batch.get('history_mask', None)
            if history_mask is not None:
                history_mask = history_mask.to(device)
            
            user_embedding = model.encode_user(user_history, history_mask)
            if not check_tensor_health(user_embedding, "用户嵌入"):
                return False
            
            # 新闻编码
            print("   -> 新闻编码阶段")
            news_input_ids = batch['news_input_ids'].to(device)
            news_sentence_positions = batch['news_sentence_positions'].to(device)
            news_attention_mask = batch.get('news_attention_mask', None)
            if news_attention_mask is not None:
                news_attention_mask = news_attention_mask.to(device)
            
            encoder_outputs = model.encode_news_body(
                news_input_ids, news_sentence_positions, news_attention_mask
            )
            if not check_tensor_health(encoder_outputs, "编码器输出"):
                return False
            
            print("✅ 前向传播检查通过")
            return True
            
    except Exception as e:
        print(f"❌ 前向传播出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_loss_computation(model, batch, device):
    """诊断损失计算过程"""
    print("\n=== 🔍 损失计算诊断 ===")
    
    model.train()
    
    try:
        # 准备输入
        user_history = batch['user_history'].to(device)
        news_input_ids = batch['news_input_ids'].to(device)
        news_sentence_positions = batch['news_sentence_positions'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        history_mask = batch.get('history_mask', None)
        if history_mask is not None:
            history_mask = history_mask.to(device)
        
        news_attention_mask = batch.get('news_attention_mask', None)
        if news_attention_mask is not None:
            news_attention_mask = news_attention_mask.to(device)
        
        print("1. 执行完整前向传播...")
        outputs = model(
            user_history=user_history,
            news_input_ids=news_input_ids,
            news_sentence_positions=news_sentence_positions,
            history_mask=history_mask,
            news_attention_mask=news_attention_mask,
            target_ids=target_ids,
            teacher_forcing_ratio=1.0
        )
        
        print("2. 检查模型输出...")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if not check_tensor_health(value, f"输出-{key}"):
                    return False
        
        print("3. 计算损失...")
        # 检查logits
        if 'logits' in outputs:
            logits = outputs['logits']
            
            # 重新整理logits和targets
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = target_ids[:, 1:].contiguous().view(-1)  # 去掉<SOS>
            
            print(f"   Logits形状: {logits_flat.shape}")
            print(f"   Targets形状: {targets_flat.shape}")
            
            if not check_tensor_health(logits_flat, "展平Logits"):
                return False
            
            if not check_tensor_health(targets_flat, "展平Targets"):
                return False
            
            # 计算损失
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
            loss = loss_fn(logits_flat, targets_flat)
            
            if not check_tensor_health(loss, "最终损失"):
                return False
            
            print(f"✅ 损失计算成功: {loss.item()}")
            return True
        else:
            print("❌ 模型输出中没有找到logits")
            return False
            
    except Exception as e:
        print(f"❌ 损失计算出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def diagnose_nan_issue():
    """完整的NaN问题诊断"""
    print("🚀 开始NaN问题诊断...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载配置
    print("\n=== 📋 加载配置 ===")
    config_path = 'configs/baseline/pens_config.yaml'
    config = load_configs(config_path)
    print("✅ 配置加载成功")
    
    # 2. 加载词汇表
    print("\n=== 📚 加载词汇表 ===")
    import json
    with open('data/processed/vocab_cache.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"✅ 词汇表加载成功，大小: {vocab_size}")
    
    # 3. 创建数据加载器（小批次测试）
    print("\n=== 📊 创建测试数据加载器 ===")
    try:
        # 直接使用硬编码的路径，避免配置问题
        train_loader, _, _ = create_data_loaders(
            train_path="data/processed/train_processed.pkl",
            valid_path="data/processed/valid_processed.pkl", 
            test_path="data/processed/test_processed.pkl",
            vocab=vocab,
            batch_size=2,  # 使用很小的批次进行测试
            num_workers=0,
            max_title_length=20,
            max_body_length=100,
            max_user_history=2
        )
        print("✅ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 创建模型
    print("\n=== 🤖 创建模型 ===")
    try:
        model = PersonalizedHeadlineGenerator(
            vocab_size=vocab_size,
            user_encoder_config=config['model']['user_encoder'],
            transformer_config=config['model']['transformer'],
            decoder_config=config['model']['decoder']
        ).to(device)
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ 模型创建成功")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 获取测试批次
    print("\n=== 🔢 获取测试批次 ===")
    try:
        batch = next(iter(train_loader))
        print(f"✅ 测试批次获取成功，批次大小: {len(batch['user_history'])}")
    except Exception as e:
        print(f"❌ 获取测试批次失败: {str(e)}")
        return False
    
    # 6. 诊断前向传播
    success = diagnose_model_forward(model, batch, device)
    if not success:
        print("❌ 前向传播诊断失败")
        return False
    
    # 7. 诊断损失计算
    success = diagnose_loss_computation(model, batch, device)
    if not success:
        print("❌ 损失计算诊断失败")
        return False
    
    print("\n🎉 所有诊断检查通过！")
    print("建议检查以下可能的问题源:")
    print("1. 学习率是否仍然过高")
    print("2. 批次大小是否过大")
    print("3. 梯度裁剪是否生效")
    print("4. 是否存在特殊的数据样本导致数值不稳定")
    
    return True

if __name__ == '__main__':
    success = diagnose_nan_issue()
    if not success:
        print("\n❌ 诊断发现问题，需要进一步调试")
        sys.exit(1)
    else:
        print("\n✅ 诊断完成，模型基础功能正常")