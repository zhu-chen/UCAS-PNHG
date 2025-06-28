# 人工智能基础大作业——个性化新闻标题生成

朱辰，潘泓锟

[toc]

## 项目简述

使用PENS数据集，根据用户的历史点击标题序列，为候选新闻生成属于这个用户的个性化标题。

本次实验PPT中提到的全部技术路线，即:

1. 基于提示工程的大模型个性化
2. 基于预训练语言模型的监督微调
3. 复现PENS中基线方法

下面将对每个技术路线进行详细介绍。

## 技术路线一：基于提示工程的大模型个性化

### 1.1 技术路线概述

## 技术路线二：基于预训练语言模型的监督微调

### 2.1 基于BART模型的微调

#### 2.1.1 BART 模型简介

BART（Bidirectional and Auto-Regressive Transformers）由 Facebook AI Research 提出 ，采用了标准的序列到序列结构，其编码器类似 BERT，为双向结构，能够充分捕捉输入文本的上下文信息；解码器类似 GPT，是从左到右的单向自回归结构 。

BART 通过对带有噪声的输入文本进行去噪重构来完成预训练，比如对输入文本执行句子乱序、文本填充等操作，然后让模型恢复原始文本。在预训练时，BART 考虑了多种噪声引入方式，包括单词掩码等。

与 BERT 独立预测掩码位置词不同，BART 以自回归的方式顺序生成。这一特性使得 BART 在处理自然语言生成任务时具备优势。预训练的 BART 模型同时具备文本表示与生成能力，适用于语言理解、文本生成等不同类型的下游任务。

#### 2.1.2 适配性分析

个性化新闻标题生成任务，需要模型根据新闻内容以及用户的个性化信息（如兴趣偏好等）生成准确且吸引人的标题。BART 模型由于其强大的文本生成能力，能够很好地适应这一任务。

一方面，BART 的双向编码器可以对新闻内容进行全面理解，捕捉其中的关键信息；另一方面，通过在预训练基础上微调，可以将用户个性化信息融入到模型的输入中，引导模型生成符合用户兴趣的标题。例如，在输入中添加用户兴趣标签与新闻内容进行拼接，模型能够在生成标题时考虑这些额外信息，从而生成更具针对性的标题。

#### 2.1.3 核心代码展示

首先通过数据集构建用户画像：

```python
def create_user_profiles(self):
        """创建用户兴趣画像，优化内存使用"""
        logger.info("开始构建用户兴趣画像...")
        
        # 为每个用户收集点击过的新闻类别
        self.user_profiles = defaultdict(lambda: defaultdict(int))
        
        # 从训练集构建用户画像 - 使用迭代器减少内存
        for _, row in tqdm.tqdm(self.train_df.iterrows(), total=len(self.train_df), desc="构建用户画像"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        category = self.news_id_to_content[news_id]['category']
                        self.user_profiles[user_id][category] += 1
```

然后对数据进行简单预处理，以降低其无效内存占用，并设置格式使其和Pytorch兼容

然后微调BART模型并进行训练:

```python
def train(self, batch_size=1, num_epochs=3, learning_rate=5e-5):
        """微调BART模型，深度优化内存使用"""
        logger.info(f"开始微调BART模型，参数配置: batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
        
        # 定义训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            save_total_limit=2,  # 只保存2个最佳模型
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),  # 启用混合精度训练
            gradient_accumulation_steps=4,  # 梯度累积，模拟更大batch_size
            dataloader_num_workers=0,  # 不使用多线程加载数据
            report_to="none",  # 不向TensorBoard等报告，减少开销
        )

        trainer.train()
```

在实际训练过程中，我们发现PENS数据集过于大，导致在本机跑数据集的过程中会出现显存不足的问题，因此我们尝试了对训练样本进行采样的方法，即只使用数据集的一部分进行训练，从而降低显存的使用，具体代码如下：

```python
def prepare_datasets(self, train_ratio=0.01, val_ratio=0.01, test_ratio=0.01):
        """准备训练、验证和测试数据集，使用比例进行采样"""
        logger.info(f"开始准备数据集，采样比例: 训练集={train_ratio*100:.1f}%, 验证集={val_ratio*100:.1f}%, 测试集={test_ratio*100:.1f}%")
        
        # 构建训练样本
        train_samples = []
        for _, row in tqdm.tqdm(self.val_df.iterrows(), total=len(self.val_df), desc="准备训练数据"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                user_profile = self.user_profiles.get(user_id, {})
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        news = self.news_id_to_content[news_id]
                        user_pref_text = self._build_user_preference_text(user_profile)
                        input_text = f"{user_pref_text} {self.user_token} {news['body']}"
                        target_text = news['title']
                        train_samples.append({
                            'user_id': user_id,
                            'news_id': news_id,
                            'input_text': input_text,
                            'target_text': target_text
                        })
```

### 2.2 基于FLAT-T5模型的微调

#### 2.2.1 FLAN-T5 模型简介
FLAN-T5 是基于 Google 的 T5（Text-To-Text Transfer Transformer）架构进一步开发的预训练语言模型。T5 将所有的自然语言处理任务统一为文本到文本的转换任务，通过大规模的文本数据预训练，在多个下游任务中表现出色。

FLAN-T5 在此基础上，通过指令调优（instruction tuning）等改进方法，增强了模型在各种任务上的性能。它在训练过程中使用了大量不同类型和领域的任务数据，包括多种语言、领域和任务类型，使模型具备了更强的泛化能力，能够根据给定的指令更好地理解和执行任务。

#### 2.2.2 适配性分析
对于个性化新闻标题生成任务，FLAN-T5 的指令调优特性使其具有独特优势。通过在大量不同任务上的训练，FLAN-T5 能够更好地理解自然语言指令。

在个性化新闻标题生成中，可以将用户个性化信息和新闻内容以指令的形式组合输入给模型，例如 “根据用户对科技领域的兴趣，为以下新闻内容生成一个吸引人的标题：[新闻内容]”，模型能够理解这种指令并生成符合要求的标题。而且其在多语言、多任务上的良好表现，意味着它能够处理多样化的新闻数据和用户需求，生成高质量的个性化标题。

#### 2.2.3 核心代码展示

类似地，首先通过数据集构建用户画像：

```python
def create_user_profiles(self):
        """创建用户兴趣画像"""
        logger.info("构建用户兴趣画像...")
        
        # 为每个用户收集点击过的新闻类别
        self.user_profiles = defaultdict(lambda: defaultdict(int))
        
        # 从训练集构建用户画像
        for _, row in tqdm(self.train_df.iterrows(), total=len(self.train_df), desc="构建用户画像"):
            user_id = row.get('UserID')
            if pd.notna(user_id):
                clicked_news_ids = str(row.get('ClicknewsID', '')).split(' ')
                for news_id in clicked_news_ids:
                    if news_id and news_id in self.news_id_to_content:
                        category = self.news_id_to_content[news_id]['category']
                        self.user_profiles[user_id][category] += 1
```

后续也是类似地首先对数据进行预处理以及采样;

然后通过微调Flan-t5模型进行训练；

```python
def train(self, batch_size=4, num_epochs=3, learning_rate=5e-5):
        """微调FLAN-T5模型，结果保存到UCAS指定目录"""
        logger.info(f"开始微调FLAN-T5模型，batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
        
        # 定义训练参数，输出到UCAS结果目录
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=f"{self.output_dir}/logs",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,
            fp16=torch.cuda.is_available()  # 使用混合精度训练加速
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        
        # 定义评估指标
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # 解码预测结果
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            # 解码目标文本
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            # 计算ROUGE分数
            result = self.rouge_metric.compute(
                predictions=decoded_preds, 
                references=decoded_labels, 
                use_stemmer=True
            )
            
            # 提取ROUGE-1, ROUGE-2, ROUGE-L的f1分数
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            
            # 添加生成的平均长度
            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
            result["gen_len"] = np.mean(prediction_lens)
            
            return {k: round(v, 4) for k, v in result.items()}
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # 开始训练
        trainer.train()
```

选择最优模型保存到相应目录下后再跑通测试集得到结果即可。

## 技术路线三：复现PENS中基线方法

### 3.1 

## 实验结果

## 小组成员分工

>朱辰：负责第一、三个技术路线的代码和报告撰写

>潘泓锟：负责第二个技术路线两个模型部分的代码和报告撰写

## 参考文献

>Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. arXiv preprint arXiv:1910.13461.

>Chung, S., Hou, D., Longpre, S., Zoph, B., Tay, Y., Fedus, W., ... & Le, Q. V. (2022). Scaling Instruction-Finetuned Language Models. arXiv preprint arXiv:2210.11416.

## 附录

### A. 数据集
https://huggingface.co/datasets/THEATLAS/PENS

### B. BART仓库链接
https://huggingface.co/facebook/bart-base

### C. FLAN-T5仓库链接
https://huggingface.co/google/flan-t5-base

### D. PENS数据集处理脚本
