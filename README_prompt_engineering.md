# 基于提示工程的个性化新闻标题生成

这是一个基于大模型API和提示工程技术的个性化新闻标题生成系统，支持多种提示策略和API提供商。



## 项目结构

```
src/prompt_engineering/
├── core/                    # 核心模块
│   ├── config.py           # 配置管理
│   ├── llm_client.py       # LLM API客户端
│   └── generator.py        # 个性化标题生成器
├── prompts/                 # 提示模板
│   └── templates.py        # 各种提示策略模板
├── utils/                   # 工具模块
│   └── data_processor.py   # 数据处理工具
└── experiments/             # 实验模块
    └── runner.py           # 实验运行器

scripts/                     # 脚本目录
└── run_prompt_engineering.py  # 主运行脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

创建并编辑 `configs/private/api_keys.private.yaml` 文件，填入你的API密钥：

```yaml
api_keys:
  siliconflow: "your_siliconflow_api_key_here"
  openai: "your_openai_api_key_here"
  zhipu: "your_zhipu_api_key_here"
```

### 3. 运行演示

```bash
python scripts/run_prompt_engineering.py demo
```

### 4. 快速测试

```bash
python scripts/run_prompt_engineering.py test --samples 5
```

## 使用指南

### 命令行接口

主要的运行脚本 `scripts/run_prompt_engineering.py` 提供了以下命令：

#### 1. 演示模式
```bash
python scripts/run_prompt_engineering.py demo
```
使用预设的示例数据演示不同策略的生成效果。

#### 2. 快速测试
```bash
python scripts/run_prompt_engineering.py test --samples 10
```
使用真实的PENS数据集进行快速测试。

#### 3. 运行实验
```bash
# 运行所有策略的比较实验
python scripts/run_prompt_engineering.py experiment --samples 50

# 运行特定策略的实验
python scripts/run_prompt_engineering.py experiment --strategies basic chain_of_thought --samples 30

# 使用特定API提供商
python scripts/run_prompt_engineering.py experiment --providers siliconflow --samples 20
```

#### 4. 列出可用策略
```bash
python scripts/run_prompt_engineering.py list
```

### 编程接口

你也可以直接在Python代码中使用：

```python
import asyncio
from src.prompt_engineering import PersonalizedTitleGenerator, template_manager

async def main():
    # 创建生成器
    generator = PersonalizedTitleGenerator()
    
    # 准备数据
    news_content = "你的新闻内容..."
    user_history = ["用户历史标题1", "用户历史标题2", ...]
    
    # 生成个性化标题
    result = await generator.generate_title(
        news_content, 
        user_history, 
        strategy="basic"  # 或其他策略
    )
    
    if result['success']:
        print(f"生成的标题: {result['generated_title']}")
    
    # 比较不同策略
    comparison = await generator.compare_strategies(news_content, user_history)
    
    await generator.close()

# 运行
asyncio.run(main())
```

## 提示策略说明

### 1. 基础策略 (basic)
直接基于用户历史偏好生成标题，适合快速生成。

### 2. 思维链推理 (chain_of_thought)
通过逐步分析用户偏好和新闻内容来生成标题，生成质量更高但耗时更长。

### 3. 角色扮演 (role_playing)
让模型扮演资深新闻编辑的角色，利用专业知识生成标题。

### 4. 少样本学习 (few_shot)
提供示例来指导模型生成，适合需要特定风格的场景。

## 配置说明

### 主配置文件 (`configs/prompt_engineering.yaml`)

- **model**: 模型相关配置
  - `primary_provider`: 主要API提供商
  - `default_model`: 默认模型名称
  - `temperature`: 生成温度
  - `max_tokens`: 最大生成长度

- **experiment**: 实验相关配置
  - `batch_size`: 批处理大小
  - `max_concurrent_requests`: 最大并发请求数
  - `max_retries`: 最大重试次数

- **data**: 数据处理配置
  - `max_history_titles`: 最大历史标题数量
  - `max_content_length`: 最大新闻内容长度

### 私有配置文件 (`configs/private/api_keys.private.yaml`)

存储API密钥等敏感信息，不会被提交到版本控制系统。

## 实验结果

实验结果会保存在 `results/prompt_engineering/` 目录下，包括：

- **详细结果**: 每个策略的完整生成结果
- **比较报告**: 不同策略的性能比较
- **统计信息**: 成功率、平均耗时等统计数据

## API提供商支持

### SiliconFlow
- 支持多种开源模型
- 成本较低
- 需要API密钥

### OpenAI
- GPT-4、GPT-3.5等模型
- 生成质量高
- 需要API密钥和可能的科学上网

## 注意事项

1. **API密钥安全**: 请妥善保管API密钥，不要提交到公开仓库
2. **费用控制**: 注意API调用费用，建议先用小样本测试
3. **网络连接**: 确保网络连接稳定，某些API可能需要科学上网
4. **数据准备**: 确保PENS数据集已正确下载到 `data/raw/` 目录

## 故障排除

### 常见问题

1. **API密钥错误**
   - 检查密钥是否正确填写在私有配置文件中
   - 确认密钥是否有效且有足够额度

2. **数据加载失败**
   - 确认PENS数据集文件存在于 `data/raw/` 目录
   - 检查文件格式是否正确

3. **网络连接问题**
   - 检查网络连接
   - 某些API可能需要配置代理

4. **内存不足**
   - 减小批处理大小
   - 减少并发请求数

## 扩展开发

### 添加新的提示策略

1. 在 `src/prompt_engineering/prompts/templates.py` 中继承 `BasePromptTemplate`
2. 实现 `format_prompt` 方法
3. 在 `PromptTemplateManager` 中注册新策略

### 添加新的API提供商

1. 在 `src/prompt_engineering/core/llm_client.py` 中继承 `BaseLLMClient`
2. 实现对应的API调用方法
3. 在 `LLMClientFactory` 中注册新提供商
