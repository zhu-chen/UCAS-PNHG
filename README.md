# UCAS-PNHG

Personalized News Headline Generation

个性化新闻标题生成

中国科学院大学人工智能基础课程大作业

根据用户的历史点击标题序列，为候选新闻生成属于这个用户的个性化标题。

数据集为[PENS](https://huggingface.co/datasets/THEATLAS/PENS)

## 项目架构(暂定)

```
UCAS-PNHG/
├── data/                    # 数据目录(过大，不放入版本控制)
│   ├── raw/                # 原始PENS数据
│   ├── processed/          # 预处理后的数据
│   └── samples/            # 数据样本
├── src/                    # 源代码目录
│   ├── data_processing/    # 数据处理模块
│   ├── models/            # 模型实现
│   ├── prompt_engineering/ # 提示工程路线
│   ├── fine_tuning/       # 微调路线
│   ├── baseline/          # 基线方法复现
│   └── evaluation/        # 评估模块
├── configs/               # 配置文件
├── scripts/               # 运行脚本
├── results/               # 实验结果
└── requirements.txt       # 依赖包
```

## 使用方法

### 环境准备
终端中运行以下命令以安装所需的Python依赖包：

```bash
pip install -r requirements.txt
```
### 下载数据

由于数据集较大，因此不放入仓库中，在运行任意实验前，请先运行`scripts/download_data.py`脚本下载数据集。

### 进行实验

见各个路线分开的README文件。

