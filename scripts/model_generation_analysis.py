#!/usr/bin/env python3
"""
模型生成案例分析器：使用训练好的模型生成标题并进行评估分析
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.baseline.models.personalized_generator import PersonalizedHeadlineGenerator
from src.baseline.evaluation.metrics import PENSEvaluator
from src.baseline.utils.logger import setup_logger

class ModelGenerationAnalyzer:
    def __init__(self, model_path, vocab_path):
        """初始化模型生成分析器"""
        self.logger = setup_logger(
            name='model_analysis',
            level=logging.INFO,
            log_file='logs/model_generation_analysis.log',
            console_output=True
        )
        
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # 创建评估器
        self.evaluator = PENSEvaluator(self.vocab)
        
        # 从检查点获取训练时保存的配置
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state = checkpoint['model_state_dict']
        saved_config = checkpoint['config']
        
        # 获取词汇表大小
        embedding_weight = model_state['user_encoder.embed.weight']
        actual_vocab_size = embedding_weight.shape[0]
        
        self.logger.info(f"检测到模型使用的词汇表大小: {actual_vocab_size}")
        
        # 构建词汇表映射
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        self.word2idx = self.vocab
        
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 使用训练时保存的配置
        model_config = saved_config.get('model', {})
        
        self.model = PersonalizedHeadlineGenerator(
            vocab_size=actual_vocab_size,
            user_encoder_config=model_config.get('user_encoder', {}),
            transformer_config=model_config.get('transformer', {}),
            decoder_config=model_config.get('decoder', {})
        )
        
        # 加载模型权重
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"模型加载完成，使用设备: {self.device}")
    
    def text_to_ids(self, text, max_length=None):
        """将文本转换为ID序列"""
        words = text.lower().split()
        ids = []
        for word in words:
            if word in self.word2idx:
                ids.append(self.word2idx[word])
            else:
                ids.append(self.word2idx.get('<UNK>', 1))
        
        # 截断或填充
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                # 用PAD填充
                pad_id = self.word2idx.get('<PAD>', 0)
                ids = ids + [pad_id] * (max_length - len(ids))
        
        return ids
    
    def ids_to_text(self, ids):
        """将ID序列转换为文本"""
        words = []
        for idx in ids:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word == '<EOS>':
                    break
                if word not in ['<PAD>', '<UNK>', '< SOS >']:
                    words.append(word)
        return ' '.join(words)
    
    def create_sample_inputs(self):
        """创建一些样本输入用于测试"""
        samples = [
            {
                'news_body': "Apple announced new iPhone with advanced camera features and longer battery life. The device will be available in multiple colors and storage options.",
                'user_history': [
                    "Apple launches new MacBook Pro with M2 chip",
                    "iPhone sales reach record high in Q3",
                    "Tesla announces new electric vehicle model"
                ],
                'original_title': "Apple Unveils Latest iPhone Model",
                'case_name': "Tech News - Apple iPhone"
            },
            {
                'news_body': "The Federal Reserve announced an interest rate increase to combat inflation. Economists predict this will impact housing market and consumer spending patterns.",
                'user_history': [
                    "Stock market volatility continues amid economic uncertainty",
                    "Inflation rates hit highest level in decades",
                    "Housing prices show signs of stabilization"
                ],
                'original_title': "Fed Raises Interest Rates Again",
                'case_name': "Finance News - Federal Reserve"
            },
            {
                'news_body': "Local basketball team wins championship after defeating rivals in overtime. The victory marks their first title in over a decade with outstanding performance.",
                'user_history': [
                    "NBA playoffs enter final rounds",
                    "Local team signs new star player",
                    "Basketball season ticket sales increase"
                ],
                'original_title': "Home Team Claims Championship Title",
                'case_name': "Sports News - Basketball Championship"
            },
            {
                'news_body': "New medical research shows promising results for cancer treatment using immunotherapy. Clinical trials demonstrate improved patient outcomes and reduced side effects.",
                'user_history': [
                    "Medical breakthrough in diabetes treatment",
                    "Hospital introduces new surgical procedures",
                    "Health insurance costs continue to rise"
                ],
                'original_title': "Cancer Treatment Shows Promise",
                'case_name': "Health News - Cancer Research"
            },
            {
                'news_body': "Climate change summit reaches agreement on carbon emission reduction targets. World leaders commit to renewable energy investments and environmental protection measures.",
                'user_history': [
                    "Solar energy adoption increases worldwide",
                    "Environmental activists protest oil drilling",
                    "Green technology stocks surge in market"
                ],
                'original_title': "Climate Summit Reaches Historic Deal",
                'case_name': "Environment News - Climate Agreement"
            }
        ]
        return samples
    
    def generate_title_for_sample(self, sample, max_length=20):
        """为单个样本生成标题"""
        try:
            # 将文本转换为ID序列
            news_body_ids = self.text_to_ids(sample['news_body'], max_length=100)
            
            # 处理用户历史
            user_history_ids = []
            for hist_text in sample['user_history']:
                hist_ids = self.text_to_ids(hist_text, max_length=20)
                user_history_ids.append(hist_ids)
            
            # 填充用户历史到固定长度
            max_history = 5
            while len(user_history_ids) < max_history:
                user_history_ids.append([0] * 20)  # 用PAD填充
            user_history_ids = user_history_ids[:max_history]
            
            # 转换为张量
            device = self.device
            news_input_ids = torch.tensor([news_body_ids], dtype=torch.long).to(device)
            user_history = torch.tensor([user_history_ids], dtype=torch.long).to(device)
            
            # 创建掩码
            news_attention_mask = (news_input_ids != 0).float()
            history_mask = (user_history != 0).float()
            
            # 生成标题
            with torch.no_grad():
                outputs = self.model.generate(
                    user_history=user_history,
                    news_input_ids=news_input_ids,
                    history_mask=history_mask,
                    news_attention_mask=news_attention_mask,
                    max_length=max_length,
                    num_beams=1
                )
                
                generated_ids = outputs['generated_ids'][0].cpu().tolist()
                
            return self.ids_to_text(generated_ids)
            
        except Exception as e:
            self.logger.error(f"生成标题失败: {str(e)}")
            return "Generation failed"
    
    def analyze_generation_cases(self, output_path, num_samples=5):
        """分析模型生成案例"""
        # 创建样本输入
        samples = self.create_sample_inputs()[:num_samples]
        
        generated_titles = []
        reference_titles = []
        case_results = []
        
        for i, sample in enumerate(samples):
            self.logger.info(f"处理案例 {i+1}: {sample['case_name']}")
            
            # 生成标题
            generated_title = self.generate_title_for_sample(sample)
            reference_title = sample['original_title']
            
            generated_titles.append(generated_title)
            reference_titles.append(reference_title)
            
            # 计算单个案例的评估指标
            rouge_scores = self.evaluator.compute_rouge_scores([generated_title], [reference_title])
            bleu_scores = self.evaluator.compute_bleu_scores([generated_title], [reference_title])
            
            case_result = {
                'case_id': i + 1,
                'case_name': sample['case_name'],
                'news_body_preview': sample['news_body'][:150] + "...",
                'user_history': sample['user_history'],
                'reference_title': reference_title,
                'generated_title': generated_title,
                'metrics': {
                    'rouge_1': rouge_scores['rouge_1'],
                    'rouge_2': rouge_scores['rouge_2'],
                    'rouge_l': rouge_scores['rouge_l'],
                    'bleu_1': bleu_scores['bleu_1'],
                    'bleu_2': bleu_scores['bleu_2'],
                    'bleu_4': bleu_scores['bleu_4']
                },
                'analysis': {
                    'generated_length': len(generated_title.split()),
                    'reference_length': len(reference_title.split()),
                    'word_overlap': self._calculate_word_overlap(generated_title, reference_title)
                }
            }
            
            case_results.append(case_result)
            
            # 打印案例结果
            print(f"\n=== 案例 {i+1}: {sample['case_name']} ===")
            print(f"新闻内容: {sample['news_body'][:100]}...")
            print(f"用户历史: {sample['user_history']}")
            print(f"参考标题: {reference_title}")
            print(f"生成标题: {generated_title}")
            print(f"ROUGE-L: {rouge_scores['rouge_l']:.3f}")
            print(f"BLEU-4: {bleu_scores['bleu_4']:.3f}")
        
        # 计算整体评估指标
        overall_metrics = self.evaluator.evaluate(generated_titles, reference_titles)
        
        # 保存结果
        results = {
            'overall_metrics': overall_metrics,
            'case_studies': case_results,
            'summary': {
                'total_cases': len(case_results),
                'avg_rouge_l': np.mean([case['metrics']['rouge_l'] for case in case_results]),
                'avg_bleu_4': np.mean([case['metrics']['bleu_4'] for case in case_results]),
                'avg_generated_length': np.mean([case['analysis']['generated_length'] for case in case_results]),
                'avg_reference_length': np.mean([case['analysis']['reference_length'] for case in case_results])
            }
        }
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"分析结果已保存到: {output_path}")
        
        # 打印总结
        print(f"\n=== 整体评估结果 ===")
        print(f"总案例数: {len(case_results)}")
        print(f"平均ROUGE-L: {results['summary']['avg_rouge_l']:.3f}")
        print(f"平均BLEU-4: {results['summary']['avg_bleu_4']:.3f}")
        print(f"平均生成长度: {results['summary']['avg_generated_length']:.1f} 词")
        print(f"平均参考长度: {results['summary']['avg_reference_length']:.1f} 词")
        
        return results
    
    def _calculate_word_overlap(self, text1, text2):
        """计算两个文本的词汇重叠"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        elif len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def load_real_test_samples(self, test_data_path, num_samples=5):
        """从实际测试数据中加载样本"""
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        
        # 随机选择样本
        import random
        random.seed(42)  # 固定随机种子以确保可重复性
        
        selected_samples = []
        sample_indices = random.sample(range(len(test_data)), min(num_samples, len(test_data)))
        
        for idx in sample_indices:
            row = test_data.iloc[idx]
            
            # 解析用户历史
            try:
                user_history = json.loads(row['user_history'])
                user_history_titles = [item['title'] for item in user_history[-3:]]  # 取最近3条
            except:
                user_history_titles = []
            
            sample = {
                'news_body': row['body'][:500],  # 截取前500字符
                'user_history': user_history_titles,
                'original_title': row['title'],
                'personalized_title': row['personalized_title'],
                'case_name': f"{row['category']} - {row['user_id']}",
                'user_id': row['user_id'],
                'news_id': row['news_id'],
                'category': row['category']
            }
            selected_samples.append(sample)
        
        return selected_samples
    
    def analyze_real_generation_cases(self, test_data_path, output_path, num_samples=5):
        """分析真实数据的模型生成案例"""
        # 加载真实测试样本
        samples = self.load_real_test_samples(test_data_path, num_samples)
        
        generated_titles = []
        reference_titles = []
        personalized_titles = []
        case_results = []
        
        for i, sample in enumerate(samples):
            self.logger.info(f"处理案例 {i+1}: {sample['case_name']}")
            
            # 生成标题
            generated_title = self.generate_title_for_sample(sample)
            reference_title = sample['original_title']
            personalized_title = sample['personalized_title']
            
            generated_titles.append(generated_title)
            reference_titles.append(reference_title)
            personalized_titles.append(personalized_title)
            
            # 计算评估指标（生成 vs 原始）
            rouge_scores_orig = self.evaluator.compute_rouge_scores([generated_title], [reference_title])
            bleu_scores_orig = self.evaluator.compute_bleu_scores([generated_title], [reference_title])
            
            # 计算评估指标（生成 vs 个性化）
            rouge_scores_pers = self.evaluator.compute_rouge_scores([generated_title], [personalized_title])
            bleu_scores_pers = self.evaluator.compute_bleu_scores([generated_title], [personalized_title])
            
            case_result = {
                'case_id': i + 1,
                'case_name': sample['case_name'],
                'user_id': sample['user_id'],
                'news_id': sample['news_id'],
                'category': sample['category'],
                'news_body_preview': sample['news_body'][:200] + "...",
                'user_history': sample['user_history'],
                'original_title': reference_title,
                'personalized_title': personalized_title,
                'model_generated_title': generated_title,
                'metrics': {
                    'vs_original': {
                        'rouge_1': rouge_scores_orig['rouge_1'],
                        'rouge_2': rouge_scores_orig['rouge_2'],
                        'rouge_l': rouge_scores_orig['rouge_l'],
                        'bleu_4': bleu_scores_orig['bleu_4']
                    },
                    'vs_personalized': {
                        'rouge_1': rouge_scores_pers['rouge_1'],
                        'rouge_2': rouge_scores_pers['rouge_2'],
                        'rouge_l': rouge_scores_pers['rouge_l'],
                        'bleu_4': bleu_scores_pers['bleu_4']
                    }
                },
                'analysis': {
                    'generated_length': len(generated_title.split()),
                    'original_length': len(reference_title.split()),
                    'personalized_length': len(personalized_title.split()),
                    'word_overlap_original': self._calculate_word_overlap(generated_title, reference_title),
                    'word_overlap_personalized': self._calculate_word_overlap(generated_title, personalized_title)
                }
            }
            
            case_results.append(case_result)
            
            # 打印案例结果
            print(f"\n=== 案例 {i+1}: {sample['case_name']} ===")
            print(f"类别: {sample['category']}")
            print(f"新闻内容: {sample['news_body'][:100]}...")
            print(f"用户历史: {sample['user_history']}")
            print(f"原始标题: {reference_title}")
            print(f"个性化标题: {personalized_title}")
            print(f"模型生成: {generated_title}")
            print(f"vs原始 - ROUGE-L: {rouge_scores_orig['rouge_l']:.3f}, BLEU-4: {bleu_scores_orig['bleu_4']:.3f}")
            print(f"vs个性化 - ROUGE-L: {rouge_scores_pers['rouge_l']:.3f}, BLEU-4: {bleu_scores_pers['bleu_4']:.3f}")
        
        # 计算整体评估指标
        overall_metrics_orig = self.evaluator.evaluate(generated_titles, reference_titles)
        overall_metrics_pers = self.evaluator.evaluate(generated_titles, personalized_titles)
        
        # 保存结果
        results = {
            'overall_metrics': {
                'vs_original': overall_metrics_orig,
                'vs_personalized': overall_metrics_pers
            },
            'case_studies': case_results,
            'summary': {
                'total_cases': len(case_results),
                'avg_rouge_l_original': np.mean([case['metrics']['vs_original']['rouge_l'] for case in case_results]),
                'avg_rouge_l_personalized': np.mean([case['metrics']['vs_personalized']['rouge_l'] for case in case_results]),
                'avg_bleu_4_original': np.mean([case['metrics']['vs_original']['bleu_4'] for case in case_results]),
                'avg_bleu_4_personalized': np.mean([case['metrics']['vs_personalized']['bleu_4'] for case in case_results]),
                'avg_generated_length': np.mean([case['analysis']['generated_length'] for case in case_results]),
                'category_distribution': {}
            }
        }
        
        # 统计类别分布
        for case in case_results:
            cat = case['category']
            if cat not in results['summary']['category_distribution']:
                results['summary']['category_distribution'][cat] = 0
            results['summary']['category_distribution'][cat] += 1
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"分析结果已保存到: {output_path}")
        
        # 打印总结
        print(f"\n=== 整体评估结果 ===")
        print(f"总案例数: {len(case_results)}")
        print(f"vs原始标题 - 平均ROUGE-L: {results['summary']['avg_rouge_l_original']:.3f}")
        print(f"vs个性化标题 - 平均ROUGE-L: {results['summary']['avg_rouge_l_personalized']:.3f}")
        print(f"vs原始标题 - 平均BLEU-4: {results['summary']['avg_bleu_4_original']:.3f}")
        print(f"vs个性化标题 - 平均BLEU-4: {results['summary']['avg_bleu_4_personalized']:.3f}")
        print(f"平均生成长度: {results['summary']['avg_generated_length']:.1f} 词")
        print(f"类别分布: {results['summary']['category_distribution']}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='模型生成案例分析器')
    parser.add_argument('--model_path', type=str, default='results/baseline/checkpoints/best_model.pth', help='模型检查点路径')
    parser.add_argument('--vocab_path', type=str, default='data/processed/vocab_cache.json', help='词汇表路径')
    parser.add_argument('--test_data_path', type=str, default='data/processed/test_processed.pkl', help='测试数据路径')
    parser.add_argument('--output_path', type=str, default='results/model_generation_analysis.json', help='输出路径')
    parser.add_argument('--num_samples', type=int, default=5, help='分析样本数量')
    parser.add_argument('--use_real_data', action='store_true', help='使用真实测试数据而不是构造样本')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = ModelGenerationAnalyzer(args.model_path, args.vocab_path)
    
    # 根据参数选择分析方式
    if args.use_real_data:
        # 使用真实测试数据进行分析
        analyzer.analyze_real_generation_cases(args.test_data_path, args.output_path, args.num_samples)
    else:
        # 使用构造样本进行分析
        analyzer.analyze_generation_cases(args.output_path, args.num_samples)


if __name__ == '__main__':
    main()