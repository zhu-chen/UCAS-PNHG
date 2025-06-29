# PENS基线方法复现

本文档详细介绍了PENS（Personalized News Headline Generation）基线方法的复现实现，基于论文《PENS: A Dataset and Generic Framework for Personalized News Headline Generation》。

## 项目概述

PENS基线方法实现了个性化新闻标题生成的最佳组合：
- **用户编码器**：NAML（Neural Attentive Multi-View Learning）
- **个性化注入策略**：F2（个性化注意力分布）
- **新闻编码器**：Transformer编码器（双层位置编码）
- **解码器**：指针生成网络（Pointer-Generator Network）

## 技术架构

### 核心组件

1. **NAML用户编码器** (`src/baseline/models/naml_encoder.py`)
   - 多视图新闻编码（标题CNN + 分类嵌入）
   - 注意力池化聚合用户历史
   - 支持点击预测预训练

2. **Transformer新闻编码器** (`src/baseline/models/generators/transformer_encoder.py`)
   - 双层位置编码（词级别 + 句子级别）
   - 多头自注意力机制（使用PyTorch内置`nn.MultiheadAttention`）
   - 层归一化和残差连接

3. **指针网络解码器** (`src/baseline/models/generators/pointer_decoder.py`)
   - LSTM解码器
   - 注意力机制
   - 指针生成机制（复制 + 生成）
   - F2个性化注入策略

4. **个性化生成器** (`src/baseline/models/personalized_generator.py`)
   - 整合所有组件
   - 实现NAML+F2最佳组合
   - 支持端到端训练

### 个性化策略

论文中提出了三种个性化注入策略，本实现重点关注表现最佳的F2策略：

- **F1**：用户嵌入初始化解码器隐状态
- **F2**：个性化注意力分布（已实现）
- **F3**：个性化生成概率

## 数据处理

### 数据预处理 (`src/baseline/data/preprocessor.py`)
- 原始PENS数据清洗和格式化
- 用户历史构建
- 文本标准化

### 数据集类 (`src/baseline/data/dataset.py`)
- 支持动态批处理
- 句子位置编码生成
- 词汇表构建和管理
- GloVe预训练嵌入加载

## 训练系统

### 训练器 (`src/baseline/training/trainer.py`)
- 多组件协同训练
- 不同学习率策略（用户编码器使用较小学习率）
- 早停和检查点管理
- 教师强制训练

### 评估系统 (`src/baseline/evaluation/metrics.py`)
- ROUGE分数（ROUGE-1, ROUGE-2, ROUGE-L）
- BLEU分数（BLEU-1至BLEU-4）
- METEOR分数
- 词汇多样性指标（Distinct-1, Distinct-2）

## 使用方法

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 确保项目结构正确
/home/zc/2025-ai/
├── src/baseline/
├── configs/baseline/
├── scripts/
└── data/
```

### 2. 数据预处理

```bash
# 预处理原始PENS数据
python scripts/run_baseline.py \
    --config configs/baseline/pens_config.yaml \
    --mode preprocess \
    --data_dir data/raw \
    --output_dir data/processed
```

### 3. 模型训练

```bash
# 训练NAML+F2模型
python scripts/run_baseline.py \
    --config configs/baseline/pens_config.yaml \
    --mode train
```

### 4. 模型评估

```bash
# 测试训练好的模型
python scripts/run_baseline.py \
    --config configs/baseline/pens_config.yaml \
    --mode eval
```

### 5. 完整流程

```bash
# 一键执行完整流程
python scripts/run_baseline.py \
    --config configs/baseline/pens_config.yaml \
    --mode full \
    --data_dir data/raw \
    --output_dir data/processed
```

## 配置说明

主要配置文件：`configs/baseline/pens_config.yaml`

### 关键参数

```yaml
model:
  # NAML用户编码器配置
  user_encoder:
    embedding_dim: 300      
    hidden_dim: 400         
    max_history_length: 50  
    num_categories: 15      
    dropout: 0.2            
  
  # Transformer编码器配置
  transformer:
    embedding_dim: 300    
    d_model: 300          
    sentence_pos_dim: 100 
    dropout: 0.2          
    max_length: 500       
  
  # 指针网络解码器配置
  decoder:
    embedding_dim: 300      # 词嵌入维度
    hidden_dim: 400         # 隐藏层维度（与Transformer输出匹配）
    num_layers: 2           # LSTM层数
    dropout: 0.2            # Dropout率
    max_decode_length: 30   # 最大解码长度
    decoder_type: 2         # F2策略
    user_size: 64           # NAML输出维度
  
  # 极简生成器配置
  generator:
    embedding_dim: 64
    hidden_dim: 128     
    num_layers: 1      
    dropout: 0.3
    max_decode_length: 15  
  
  # 词嵌入配置
  word_embedding:
    dim: 128            
    pretrained: false

```

## 实验结果

模型在PENS数据集上的预期性能指标：

| 指标 | 分数范围 | 说明 |
|------|----------|------|
| ROUGE-1 | 0.25-0.35 | 单词级别重叠 |
| ROUGE-2 | 0.15-0.25 | 双词级别重叠 |
| ROUGE-L | 0.20-0.30 | 最长公共子序列 |
| BLEU-4 | 0.10-0.20 | 4-gram精确匹配 |


## 代码优化

相比原始实现，本项目有以下优化：

1. **统一的检查点管理**：支持训练恢复和最佳模型保存
2. **灵活的配置系统**：YAML配置文件管理所有参数


