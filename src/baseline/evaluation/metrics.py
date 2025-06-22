"""
PENS评估指标
实现个性化新闻标题生成的评估指标，包括ROUGE、BLEU等
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import Counter
import re

logger = logging.getLogger(__name__)


class PENSEvaluator:
    """PENS模型评估器"""
    
    def __init__(self, vocab: Dict[str, int]):
        """
        Args:
            vocab: 词汇表
        """
        self.vocab = vocab
        self.id2word = {v: k for k, v in vocab.items()}
        
        # 特殊token
        self.pad_id = vocab.get("<PAD>", 0)
        self.unk_id = vocab.get("<UNK>", 1)
        self.sos_id = vocab.get("<SOS>", 2)
        self.eos_id = vocab.get("<EOS>", 3)
    
    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        综合评估
        
        Args:
            predictions: 预测结果列表
            references: 参考答案列表
        
        Returns:
            评估指标字典
        """
        if len(predictions) != len(references):
            raise ValueError(f"预测和参考数量不匹配: {len(predictions)} vs {len(references)}")
        
        # 计算各项指标
        rouge_scores = self.compute_rouge_scores(predictions, references)
        bleu_scores = self.compute_bleu_scores(predictions, references)
        meteor_score = self.compute_meteor_score(predictions, references)
        
        # 长度统计
        length_stats = self.compute_length_statistics(predictions, references)
        
        # 词汇多样性
        diversity_stats = self.compute_diversity_statistics(predictions)
        
        # 整合所有指标
        metrics = {
            'rouge_1': rouge_scores['rouge_1'],
            'rouge_2': rouge_scores['rouge_2'],
            'rouge_l': rouge_scores['rouge_l'],
            'bleu_1': bleu_scores['bleu_1'],
            'bleu_2': bleu_scores['bleu_2'],
            'bleu_3': bleu_scores['bleu_3'],
            'bleu_4': bleu_scores['bleu_4'],
            'meteor': meteor_score,
            'avg_pred_length': length_stats['avg_pred_length'],
            'avg_ref_length': length_stats['avg_ref_length'],
            'distinct_1': diversity_stats['distinct_1'],
            'distinct_2': diversity_stats['distinct_2']
        }
        
        # 计算综合分数（主要基于ROUGE-L）
        metrics['rouge'] = rouge_scores['rouge_l']
        metrics['bleu'] = bleu_scores['bleu_4']
        
        return metrics
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数"""
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            # ROUGE-1
            rouge_1 = self._compute_rouge_n(pred_tokens, ref_tokens, n=1)
            rouge_1_scores.append(rouge_1)
            
            # ROUGE-2
            rouge_2 = self._compute_rouge_n(pred_tokens, ref_tokens, n=2)
            rouge_2_scores.append(rouge_2)
            
            # ROUGE-L
            rouge_l = self._compute_rouge_l(pred_tokens, ref_tokens)
            rouge_l_scores.append(rouge_l)
        
        return {
            'rouge_1': np.mean(rouge_1_scores),
            'rouge_2': np.mean(rouge_2_scores),
            'rouge_l': np.mean(rouge_l_scores)
        }
    
    def compute_bleu_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算BLEU分数"""
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            # 计算不同n-gram的BLEU
            for n in range(1, 5):
                bleu_n = self._compute_bleu_n(pred_tokens, [ref_tokens], n)
                if n == 1:
                    bleu_1_scores.append(bleu_n)
                elif n == 2:
                    bleu_2_scores.append(bleu_n)
                elif n == 3:
                    bleu_3_scores.append(bleu_n)
                elif n == 4:
                    bleu_4_scores.append(bleu_n)
        
        return {
            'bleu_1': np.mean(bleu_1_scores),
            'bleu_2': np.mean(bleu_2_scores),
            'bleu_3': np.mean(bleu_3_scores),
            'bleu_4': np.mean(bleu_4_scores)
        }
    
    def compute_meteor_score(self, predictions: List[str], references: List[str]) -> float:
        """计算METEOR分数（简化版本）"""
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._tokenize(pred))
            ref_tokens = set(self._tokenize(ref))
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                meteor_scores.append(1.0)
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                meteor_scores.append(0.0)
            else:
                # 简化的METEOR计算（仅考虑精确匹配）
                matches = len(pred_tokens & ref_tokens)
                precision = matches / len(pred_tokens)
                recall = matches / len(ref_tokens)
                
                if precision + recall == 0:
                    meteor = 0.0
                else:
                    meteor = 2 * precision * recall / (precision + recall)
                
                meteor_scores.append(meteor)
        
        return np.mean(meteor_scores)
    
    def compute_length_statistics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算长度统计"""
        pred_lengths = [len(self._tokenize(pred)) for pred in predictions]
        ref_lengths = [len(self._tokenize(ref)) for ref in references]
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'std_pred_length': np.std(pred_lengths),
            'std_ref_length': np.std(ref_lengths)
        }
    
    def compute_diversity_statistics(self, predictions: List[str]) -> Dict[str, float]:
        """计算词汇多样性统计"""
        all_tokens = []
        all_bigrams = []
        
        for pred in predictions:
            tokens = self._tokenize(pred)
            all_tokens.extend(tokens)
            
            # 构建bigrams
            for i in range(len(tokens) - 1):
                all_bigrams.append((tokens[i], tokens[i+1]))
        
        # Distinct-1: 不重复的unigram数量占总unigram数量的比例
        if len(all_tokens) == 0:
            distinct_1 = 0.0
        else:
            distinct_1 = len(set(all_tokens)) / len(all_tokens)
        
        # Distinct-2: 不重复的bigram数量占总bigram数量的比例
        if len(all_bigrams) == 0:
            distinct_2 = 0.0
        else:
            distinct_2 = len(set(all_bigrams)) / len(all_bigrams)
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens))
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if not text:
            return []
        
        # 简单的空格分词
        tokens = text.lower().strip().split()
        
        # 过滤特殊token
        filtered_tokens = []
        for token in tokens:
            if token not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _compute_rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """计算ROUGE-N分数"""
        if len(ref_tokens) < n:
            return 0.0
        
        # 构建n-gram
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)
        
        if len(ref_ngrams) == 0:
            return 0.0
        
        # 计算重叠
        overlap = sum((pred_ngrams & ref_ngrams).values())
        total_ref_ngrams = sum(ref_ngrams.values())
        
        return overlap / total_ref_ngrams
    
    def _compute_rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """计算ROUGE-L分数（基于最长公共子序列）"""
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        # 计算最长公共子序列长度
        lcs_length = self._lcs_length(pred_tokens, ref_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        # 计算ROUGE-L
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _compute_bleu_n(self, pred_tokens: List[str], ref_tokens_list: List[List[str]], n: int) -> float:
        """计算BLEU-N分数"""
        if len(pred_tokens) < n:
            return 0.0
        
        # 构建预测的n-gram
        pred_ngrams = self._get_ngrams(pred_tokens, n)
        
        if len(pred_ngrams) == 0:
            return 0.0
        
        # 计算与所有参考的最大重叠
        max_overlap = 0
        for ref_tokens in ref_tokens_list:
            if len(ref_tokens) >= n:
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                overlap = sum((pred_ngrams & ref_ngrams).values())
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap / sum(pred_ngrams.values())
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """获取n-gram计数"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """计算最长公共子序列长度"""
        m, n = len(seq1), len(seq2)
        
        # 创建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def decode_ids_to_tokens(self, token_ids: torch.Tensor) -> List[str]:
        """将token ID解码为词汇"""
        tokens = []
        for token_id in token_ids:
            if isinstance(token_id, torch.Tensor):
                token_id = token_id.item()
            
            if token_id == self.eos_id:  # EOS token
                break
            elif token_id in [self.pad_id, self.sos_id]:  # 跳过特殊token
                continue
            elif token_id in self.id2word:
                tokens.append(self.id2word[token_id])
            else:
                tokens.append(self.id2word[self.unk_id])  # UNK token
        
        return tokens
    
    def decode_batch_ids_to_text(self, batch_ids: torch.Tensor) -> List[str]:
        """批量解码token ID为文本"""
        batch_size = batch_ids.size(0)
        texts = []
        
        for i in range(batch_size):
            tokens = self.decode_ids_to_tokens(batch_ids[i])
            text = ' '.join(tokens)
            texts.append(text)
        
        return texts


class PersonalizationEvaluator:
    """个性化效果评估器"""
    
    def __init__(self):
        self.user_predictions = {}
        self.user_references = {}
    
    def add_user_predictions(self, user_id: str, predictions: List[str], references: List[str]):
        """添加用户预测结果"""
        if user_id not in self.user_predictions:
            self.user_predictions[user_id] = []
            self.user_references[user_id] = []
        
        self.user_predictions[user_id].extend(predictions)
        self.user_references[user_id].extend(references)
    
    def evaluate_personalization_effect(self) -> Dict[str, float]:
        """评估个性化效果"""
        if not self.user_predictions:
            return {}
        
        # 计算每个用户的ROUGE分数
        user_rouge_scores = {}
        evaluator = PENSEvaluator({})  # 临时evaluator
        
        for user_id in self.user_predictions:
            preds = self.user_predictions[user_id]
            refs = self.user_references[user_id]
            
            if len(preds) > 0 and len(refs) > 0:
                rouge_scores = evaluator.compute_rouge_scores(preds, refs)
                user_rouge_scores[user_id] = rouge_scores['rouge_l']
        
        if not user_rouge_scores:
            return {}
        
        # 计算个性化指标
        scores = list(user_rouge_scores.values())
        
        return {
            'avg_user_rouge': np.mean(scores),
            'std_user_rouge': np.std(scores),
            'min_user_rouge': np.min(scores),
            'max_user_rouge': np.max(scores),
            'num_users': len(user_rouge_scores)
        }


def evaluate_model_predictions(
    predictions_file: str,
    references_file: str,
    vocab: Dict[str, int]
) -> Dict[str, float]:
    """从文件评估模型预测结果"""
    # 加载预测和参考
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = [line.strip() for line in f]
    
    with open(references_file, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f]
    
    # 创建评估器并评估
    evaluator = PENSEvaluator(vocab)
    metrics = evaluator.evaluate(predictions, references)
    
    return metrics