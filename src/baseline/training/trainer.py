"""
PENS基线模型训练器
实现NAML+F2个性化新闻标题生成模型的训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from pathlib import Path
import json
from tqdm import tqdm
import time

from ..models.personalized_generator import create_naml_f2_generator
from ..data.dataset import create_data_loaders, build_vocab, load_glove_embeddings
from ..evaluation.metrics import PENSEvaluator
from ..utils.checkpoints import CheckpointManager
from ..utils.logger import setup_logger

logger = logging.getLogger(__name__)


class PENSTrainer:
    """PENS模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 训练配置
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 增加详细日志
        logger.info(f"=== 初始化PENSTrainer ===")
        logger.info(f"使用设备: {self.device}")
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"当前GPU: {torch.cuda.current_device()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        logger.info(f"配置参数概要:")
        logger.info(f"  批次大小: {config.get('batch_size', 32)}")
        logger.info(f"  训练轮数: {config.get('num_epochs', 10)}")
        logger.info(f"  学习率: {config.get('optimizer', {}).get('lr', 1e-4)}")
        logger.info(f"  数据路径: {config.get('data', {})}")
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.evaluator = None
        self.checkpoint_manager = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = 0.0
        self.patience_counter = 0
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding token
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'valid_loss': [],
            'rouge_scores': [],
            'bleu_scores': []
        }
        
        logger.info("PENSTrainer初始化完成")
    
    def setup(self):
        """初始化训练环境"""
        logger.info("=== 开始设置训练环境 ===")
        
        # 设置随机种子
        logger.info("步骤 1/6: 设置随机种子")
        self._set_seed(self.config.get('seed', 42))
        logger.info("随机种子设置完成")
        
        # 设置日志
        logger.info("步骤 2/6: 设置日志系统")
        setup_logger(self.config.get('log_level', 'INFO'))
        logger.info("日志系统设置完成")
        
        # 构建词汇表
        logger.info("步骤 3/6: 构建或加载词汇表")
        try:
            vocab = self._build_or_load_vocab()
            logger.info(f"词汇表处理完成，词汇表大小: {len(vocab)}")
        except Exception as e:
            logger.error(f"构建词汇表时出错: {str(e)}")
            raise e
        
        # 创建数据加载器
        logger.info("步骤 4/6: 创建数据加载器")
        try:
            self._create_data_loaders(vocab)
            logger.info("数据加载器创建完成")
        except Exception as e:
            logger.error(f"创建数据加载器时出错: {str(e)}")
            raise e
        
        # 创建模型
        logger.info("步骤 5/6: 创建模型")
        try:
            self._create_model(vocab)
            logger.info("模型创建完成")
        except Exception as e:
            logger.error(f"创建模型时出错: {str(e)}")
            raise e
        
        # 创建优化器和调度器
        logger.info("步骤 6/6: 创建优化器和调度器")
        try:
            self._create_optimizer()
            logger.info("优化器和调度器创建完成")
        except Exception as e:
            logger.error(f"创建优化器时出错: {str(e)}")
            raise e
        
        # 创建评估器和检查点管理器（简化版本，避免依赖问题）
        logger.info("创建评估器和检查点管理器...")
        try:
            self._create_evaluator(vocab)
            self._create_checkpoint_manager()
            logger.info("评估器和检查点管理器创建完成")
        except Exception as e:
            logger.warning(f"创建评估器或检查点管理器时出错: {str(e)}，将使用简化版本")
            self.evaluator = None
            self.checkpoint_manager = None
        
        logger.info(f"=== 训练环境初始化完成，使用设备: {self.device} ===")
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _build_or_load_vocab(self) -> Dict[str, int]:
        """构建或加载词汇表"""
        logger.info("开始处理词汇表...")
        vocab_path = self.config.get('vocab_path')
        logger.info(f"词汇表路径配置: {vocab_path}")
        
        if vocab_path and Path(vocab_path).exists():
            logger.info(f"发现已存在的词汇表文件: {vocab_path}")
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = json.load(f)
                logger.info(f"成功从文件加载词汇表，词汇数量: {len(vocab)}")
                return vocab
            except Exception as e:
                logger.error(f"加载词汇表文件失败: {str(e)}")
                raise e
        else:
            logger.info("未找到已存在的词汇表，开始构建新词汇表")
            try:
                data_paths = [
                    self.config['data']['train_path'],
                    self.config['data']['valid_path']
                ]
                logger.info(f"将从以下数据文件构建词汇表: {data_paths}")
                
                # 检查数据文件是否存在
                for path in data_paths:
                    if not Path(path).exists():
                        logger.error(f"数据文件不存在: {path}")
                        raise FileNotFoundError(f"数据文件不存在: {path}")
                    else:
                        logger.info(f"确认数据文件存在: {path}")
                
                logger.info("开始调用build_vocab函数...")
                vocab = build_vocab(
                    data_paths=data_paths,
                    vocab_size=self.config.get('vocab_size', 50000),
                    min_freq=self.config.get('min_word_freq', 2)
                )
                logger.info(f"词汇表构建完成，词汇数量: {len(vocab)}")
                
                # 保存词汇表
                if vocab_path:
                    logger.info(f"保存词汇表到: {vocab_path}")
                    os.makedirs(Path(vocab_path).parent, exist_ok=True)
                    with open(vocab_path, 'w', encoding='utf-8') as f:
                        json.dump(vocab, f, ensure_ascii=False, indent=2)
                    logger.info("词汇表保存完成")
                
                return vocab
            except Exception as e:
                logger.error(f"构建词汇表过程中出错: {str(e)}")
                raise e
    
    def _create_data_loaders(self, vocab: Dict[str, int]):
        """创建数据加载器"""
        logger.info("=== 开始创建数据加载器 ===")
        
        # 记录配置信息
        train_path = self.config['data']['train_path']
        valid_path = self.config['data']['valid_path'] 
        test_path = self.config['data']['test_path']
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        logger.info(f"数据路径配置:")
        logger.info(f"  训练集: {train_path}")
        logger.info(f"  验证集: {valid_path}")
        logger.info(f"  测试集: {test_path}")
        logger.info(f"批次大小: {batch_size}")
        logger.info(f"工作进程数: {num_workers}")
        logger.info(f"词汇表大小: {len(vocab)}")
        
        # 检查数据文件是否存在
        logger.info("检查数据文件存在性...")
        for name, path in [("训练集", train_path), ("验证集", valid_path), ("测试集", test_path)]:
            if not Path(path).exists():
                logger.error(f"{name}文件不存在: {path}")
                raise FileNotFoundError(f"{name}文件不存在: {path}")
            else:
                # 获取文件大小信息
                file_size = Path(path).stat().st_size / (1024 * 1024)  # MB
                logger.info(f"  ✓ {name}文件存在: {path} (大小: {file_size:.1f} MB)")
        
        # 检查数据加载函数是否存在
        logger.info("检查create_data_loaders函数...")
        try:
            from ..data.dataset import create_data_loaders
            logger.info("create_data_loaders函数导入成功")
        except ImportError as e:
            logger.error(f"无法导入create_data_loaders函数: {str(e)}")
            raise e
        
        try:
            logger.info("开始调用create_data_loaders函数...")
            start_time = time.time()
            
            # 尝试加载一小批数据先测试
            logger.info("创建数据加载器实例...")
            self.train_loader, self.valid_loader, self.test_loader = create_data_loaders(
                train_path=train_path,
                valid_path=valid_path,
                test_path=test_path,
                vocab=vocab,
                batch_size=batch_size,
                num_workers=num_workers,
                max_title_length=self.config.get('max_title_length', 30),
                max_body_length=self.config.get('max_body_length', 500),
                max_user_history=self.config.get('max_user_history', 50)
            )
            
            end_time = time.time()
            logger.info(f"数据加载器创建完成，耗时: {end_time - start_time:.2f}秒")
            
            logger.info(f"数据加载器信息:")
            logger.info(f"  训练集批次数: {len(self.train_loader)}")
            logger.info(f"  验证集批次数: {len(self.valid_loader)}")
            logger.info(f"  测试集批次数: {len(self.test_loader)}")
            
            # 测试加载第一个批次
            logger.info("测试加载第一个训练批次...")
            try:
                first_batch = next(iter(self.train_loader))
                logger.info(f"第一个批次键: {list(first_batch.keys())}")
                for key, value in first_batch.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f"  {key}: {value.shape} ({value.dtype})")
                    else:
                        logger.info(f"  {key}: {type(value)}")
                logger.info("第一个批次加载测试成功")
            except Exception as e:
                logger.error(f"第一个批次加载测试失败: {str(e)}")
                import traceback
                logger.error(f"错误堆栈: {traceback.format_exc()}")
                raise e
            
        except Exception as e:
            logger.error(f"创建数据加载器失败: {str(e)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            raise e
    
    def _create_model(self, vocab: Dict[str, int]):
        """创建模型"""
        model_config = self.config.get('model', {})
        
        self.model = create_naml_f2_generator(
            vocab_size=len(vocab),
            config=model_config
        )
        
        # 加载预训练词嵌入
        if self.config.get('pretrained_embeddings'):
            embeddings = load_glove_embeddings(
                self.config['pretrained_embeddings'],
                vocab,
                embedding_dim=model_config.get('user_encoder', {}).get('embedding_dim', 300)
            )
            self.model.load_pretrained_embeddings(embeddings)
            logger.info("已加载预训练词嵌入")
        
        self.model.to(self.device)
        
        # 打印模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")
    
    def _create_optimizer(self):
        """创建优化器和学习率调度器"""
        optimizer_config = self.config.get('optimizer', {})
        
        # 不同组件使用不同学习率
        param_groups = []
        
        # 用户编码器参数（较小学习率）
        user_encoder_params = list(self.model.user_encoder.parameters())
        param_groups.append({
            'params': user_encoder_params,
            'lr': optimizer_config.get('user_encoder_lr', 1e-5)
        })
        
        # 其他参数（标准学习率）
        other_params = []
        for name, param in self.model.named_parameters():
            if not name.startswith('user_encoder'):
                other_params.append(param)
        
        param_groups.append({
            'params': other_params,
            'lr': optimizer_config.get('lr', 1e-4)
        })
        
        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            eps=optimizer_config.get('eps', 1e-8)
        )
        
        # 学习率调度器
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'linear')
        
        if scheduler_type == 'linear':
            num_training_steps = len(self.train_loader) * self.config.get('num_epochs', 10)
            num_warmup_steps = int(num_training_steps * scheduler_config.get('warmup_ratio', 0.1))
            
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 10)
            )
        else:
            self.scheduler = None
    
    def _create_evaluator(self, vocab: Dict[str, int]):
        """创建评估器"""
        self.evaluator = PENSEvaluator(vocab)
    def _create_checkpoint_manager(self):
        """创建检查点管理器"""
        checkpoint_config = self.config.get('checkpoint', {})
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_config.get('dir', './checkpoints'),
            max_checkpoints=checkpoint_config.get('max_keep', 5)
        )
    
    def train(self):
        """开始训练"""
        logger.info("开始训练PENS模型")
        
        num_epochs = self.config.get('num_epochs', 10)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self._train_epoch()
            
            # 验证
            valid_loss, valid_metrics = self._validate()
            
            # 更新训练历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['valid_loss'].append(valid_loss)
            self.train_history['rouge_scores'].append(valid_metrics.get('rouge', 0.0))
            self.train_history['bleu_scores'].append(valid_metrics.get('bleu', 0.0))
            
            # 记录日志
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
                f"ROUGE: {valid_metrics.get('rouge', 0.0):.4f}, "
                f"BLEU: {valid_metrics.get('bleu', 0.0):.4f}"
            )
            
            # 早停检查
            current_score = valid_metrics.get('rouge', 0.0)
            if current_score > self.best_score:
                self.best_score = current_score
                self.patience_counter = 0
                
                # 保存最佳模型
                self.checkpoint_manager.save_checkpoint(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'best_score': self.best_score,
                        'config': self.config
                    },
                    is_best=True
                )
            else:
                self.patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.checkpoint_manager.save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_score': self.best_score,
                    'config': self.config
                })
            
            # 早停
            if self.patience_counter >= self.config.get('patience', 3):
                logger.info(f"验证性能连续 {self.patience_counter} 轮未提升，提前停止训练")
                break
        
        logger.info("训练完成")
    
    def _train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            
            outputs = self.model(
                user_history=batch['user_history'],
                history_mask=batch['history_mask'],
                history_categories=batch['history_categories'],
                news_input_ids=batch['news_input_ids'],
                news_sentence_positions=batch['news_sentence_positions'],
                news_attention_mask=batch['news_attention_mask'],
                target_ids=batch['target_ids'],
                teacher_forcing_ratio=self.config.get('teacher_forcing_ratio', 1.0)
            )
            
            # 计算损失
            loss = self._calculate_loss(outputs, batch)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', 0) > 0:
                clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        return total_loss / num_batches
    
    def _validate(self) -> Tuple[float, Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(
                    user_history=batch['user_history'],
                    history_mask=batch['history_mask'],
                    history_categories=batch['history_categories'],
                    news_input_ids=batch['news_input_ids'],
                    news_sentence_positions=batch['news_sentence_positions'],
                    news_attention_mask=batch['news_attention_mask'],
                    target_ids=batch['target_ids'],
                    teacher_forcing_ratio=0.0  # 验证时不使用教师强制
                )
                
                # 计算损失
                loss = self._calculate_loss(outputs, batch)
                total_loss += loss.item()
                
                # 解码预测结果
                batch_predictions = self._decode_predictions(outputs['final_dists'])
                batch_references = self._decode_references(batch['target_ids'])
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        # 计算评估指标
        metrics = self.evaluator.evaluate(predictions, references)
        
        return total_loss / len(self.valid_loader), metrics
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失"""
        final_dists = outputs['final_dists']  # [batch_size, seq_len, vocab_size]
        target_ids = batch['target_ids'][:, 1:]  # 去掉SOS token
        target_mask = batch.get('target_mask', None)
        if target_mask is not None:
            target_mask = target_mask[:, 1:]  # 对应的掩码
        
        batch_size, seq_len, vocab_size = final_dists.size()
        target_batch_size, target_seq_len = target_ids.size()
        
        # 确保序列长度匹配
        min_seq_len = min(seq_len, target_seq_len)
        final_dists = final_dists[:, :min_seq_len, :]  # 截取到最小长度
        target_ids = target_ids[:, :min_seq_len]       # 截取到最小长度
        
        # 重塑张量以适应CrossEntropyLoss
        final_dists = final_dists.contiguous().view(-1, vocab_size)
        target_ids = target_ids.contiguous().view(-1)
        
        loss = self.criterion(final_dists, target_ids)
        
        return loss
    
    def _decode_predictions(self, final_dists: torch.Tensor) -> List[str]:
        """解码预测结果"""
        batch_size, seq_len, vocab_size = final_dists.size()
        
        # 获取预测的token ID
        pred_ids = torch.argmax(final_dists, dim=-1)  # [batch_size, seq_len]
        
        predictions = []
        for i in range(batch_size):
            pred_tokens = []
            for j in range(seq_len):
                token_id = pred_ids[i, j].item()
                if token_id == 3:  # EOS token
                    break
                elif token_id > 3:  # 跳过特殊token
                    # 这里需要ID到词的映射，简化处理
                    pred_tokens.append(str(token_id))
            
            predictions.append(' '.join(pred_tokens))
        
        return predictions
    
    def _decode_references(self, target_ids: torch.Tensor) -> List[str]:
        """解码参考答案"""
        batch_size, seq_len = target_ids.size()
        
        references = []
        for i in range(batch_size):
            ref_tokens = []
            for j in range(seq_len):
                token_id = target_ids[i, j].item()
                if token_id == 3:  # EOS token
                    break
                elif token_id > 3:  # 跳过特殊token
                    ref_tokens.append(str(token_id))
            
            references.append(' '.join(ref_tokens))
        
        return references
    
    def test(self):
        """测试模型"""
        logger.info("开始测试模型")
        
        # 加载最佳模型
        best_checkpoint = self.checkpoint_manager.load_best_checkpoint()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint['model_state_dict'])
            logger.info("已加载最佳模型")
        
        self.model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    user_history=batch['user_history'],
                    history_mask=batch['history_mask'],
                    history_categories=batch['history_categories'],
                    news_input_ids=batch['news_input_ids'],
                    news_sentence_positions=batch['news_sentence_positions'],
                    news_attention_mask=batch['news_attention_mask'],
                    teacher_forcing_ratio=0.0
                )
                
                batch_predictions = self._decode_predictions(outputs['final_dists'])
                batch_references = self._decode_references(batch['target_ids'])
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        # 评估
        test_metrics = self.evaluator.evaluate(predictions, references)
        
        logger.info("测试结果:")
        for metric, score in test_metrics.items():
            logger.info(f"{metric}: {score:.4f}")
        
        return test_metrics
    
    def save_training_history(self, save_path: str):
        """保存训练历史"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info(f"训练历史已保存到 {save_path}")


def create_trainer(config_path: str) -> PENSTrainer:
    """创建训练器"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    return PENSTrainer(config)