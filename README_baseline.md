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

### 4. 模型测试

```bash
# 测试训练好的模型
python scripts/run_baseline.py \
    --config configs/baseline/pens_config.yaml \
    --mode test
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
# 模型架构
model:
  user_encoder:
    hidden_dim: 256           # 用户嵌入维度
    max_history_length: 50    # 最大用户历史长度
  
  transformer:
    d_model: 512              # Transformer隐藏维度
    num_heads: 8              # 注意力头数
    num_layers: 6             # Transformer层数
  
  decoder:
    hidden_dim: 512           # 解码器隐藏维度
    max_decode_length: 30     # 最大生成长度

# 训练参数
num_epochs: 20
batch_size: 32
optimizer:
  lr: 1e-4                   # 主学习率
  user_encoder_lr: 1e-5      # 用户编码器学习率
```

## 实验结果

模型在PENS数据集上的预期性能指标：

| 指标 | 分数范围 | 说明 |
|------|----------|------|
| ROUGE-1 | 0.25-0.35 | 单词级别重叠 |
| ROUGE-2 | 0.15-0.25 | 双词级别重叠 |
| ROUGE-L | 0.20-0.30 | 最长公共子序列 |
| BLEU-4 | 0.10-0.20 | 4-gram精确匹配 |

## 技术特点

1. **模块化设计**：各组件独立实现，便于扩展和修改
2. **高效实现**：使用PyTorch内置API，性能优化
3. **完整流程**：从数据预处理到模型评估的完整pipeline
4. **论文复现**：严格按照PENS论文实现最佳方法组合

## 代码优化

相比原始实现，本项目有以下优化：

1. **使用PyTorch内置多头注意力**：`nn.MultiheadAttention`替代自定义实现
2. **统一的检查点管理**：支持训练恢复和最佳模型保存
3. **详细的评估指标**：多维度性能评估
4. **灵活的配置系统**：YAML配置文件管理所有参数

## 扩展方向

1. **其他个性化策略**：实现F1和F3策略的对比实验
2. **用户编码器变体**：尝试其他用户建模方法
3. **解码器改进**：集成更先进的生成模型
4. **评估增强**：添加人工评估和个性化效果分析

## 参考文献

```
@inproceedings{luo-etal-2021-pens,
    title = "PENS: A Dataset and Generic Framework for Personalized News Headline Generation",
    author = "Luo, Luling and Ao, Xiang and Pan, Feiyang and Song, Yanling and He, Qing",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    year = "2021",
    pages = "82--92"
}
```

## 故障排除

### 常见问题

1. **内存不足**：减小`batch_size`或`max_body_length`
2. **CUDA错误**：检查GPU兼容性，或设置`device: "cpu"`
3. **数据路径错误**：确保数据文件存在于指定路径
4. **依赖缺失**：安装requirements.txt中的所有依赖

### 性能调优

1. **训练速度**：增加`num_workers`，使用混合精度训练
2. **内存优化**：使用梯度检查点，调整序列长度
3. **收敛改进**：调整学习率，使用预训练词嵌入

## 项目状态

- ✅ 核心模型实现完成
- ✅ 训练系统完成
- ✅ 评估系统完成
- ✅ 数据处理完成
- ✅ 配置和脚本完成
- ⏳ 实验验证进行中

本实现为PENS基线方法的完整复现，可直接用于个性化新闻标题生成的研究和应用。

