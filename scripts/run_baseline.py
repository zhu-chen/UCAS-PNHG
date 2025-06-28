#!/usr/bin/env python3
"""
PENS基线模型训练脚本
使用稳定训练器确保数值稳定性
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
import yaml

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baseline.models.personalized_generator import PersonalizedHeadlineGenerator
from src.baseline.data.dataset import PENSDataset
from src.baseline.data.preprocessor import PENSPreprocessor
from src.baseline.training.trainer import PENSTrainer
from src.baseline.utils.config import Config  # 修复导入
from src.baseline.utils.logger import setup_logger


def load_config(config_path: str):
    """简单的配置加载函数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='PENS基线模型训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train', help='运行模式')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID')
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        print(f"使用GPU: {device}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger('trainer', logging.INFO, 'logs/training.log')
    logger.info(f"开始训练，配置文件: {args.config}")
    logger.info(f"使用设备: {device}")
    
    # 确保输出目录存在
    os.makedirs('results/baseline/checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # 加载和预处理数据
        logger.info("加载数据...")
        preprocessor = PENSPreprocessor(config['data'])
        logger.info("预处理数据...")
        # 加载训练数据
        train_data = preprocessor.load_processed_data('data/processed/train_processed.pkl')
        logger.info(f"训练数据加载完成，包含 {len(train_data)} 条记录")
        val_data = preprocessor.load_processed_data('data/processed/valid_processed.pkl')
        logger.info(f"验证数据加载完成，包含 {len(val_data)} 条记录")

        # 创建数据集
        # 如果train_data是DataFrame，需要先保存为文件或者传递文件路径
        if hasattr(train_data, 'to_pickle'):  # 检查是否为DataFrame
            # 创建临时文件路径
            temp_train_path = "data/temp/train_data.pkl"
            os.makedirs("data/temp", exist_ok=True)
            train_data.to_pickle(temp_train_path)
            train_dataset = PENSDataset(temp_train_path, preprocessor.vocab, mode='train')
        else:
            # 如果已经是路径字符串
            train_dataset = PENSDataset(train_data, preprocessor.vocab, mode='train')

        val_dataset = PENSDataset(val_data, preprocessor.vocab, mode='val')
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        logger.info(f"词汇表大小: {len(preprocessor.vocab)}")
        
        # 创建模型
        logger.info("创建模型...")
        model = PersonalizedHeadlineGenerator(
            vocab_size=len(preprocessor.vocab),
            user_encoder_config=config['model']['user_encoder'],
            transformer_config=config['model']['transformer'],
            decoder_config=config['model']['decoder']
        )
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型总参数数: {total_params:,}")
        logger.info(f"可训练参数数: {trainable_params:,}")
        
        # 加载预训练词嵌入（如果有）
        embeddings_path = config.get('embeddings_path')
        if embeddings_path and os.path.exists(embeddings_path):
            logger.info(f"加载预训练词嵌入: {embeddings_path}")
            embeddings = torch.load(embeddings_path)
            model.load_pretrained_embeddings(embeddings)
        
        # 恢复训练（如果指定）
        start_epoch = 0
        if args.resume and os.path.exists(args.resume):
            logger.info(f"从检查点恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        
        # 创建训练器
        logger.info("初始化训练器...")
        # PENSTrainer只接受config参数，模型会在内部创建
        trainer = PENSTrainer(config)
        
        if args.mode == 'train':
            # 设置训练环境
            trainer.setup()
            # 开始训练
            logger.info("开始训练...")
            trainer.train()
            logger.info("训练完成！")
            
        elif args.mode == 'eval':
            # 评估模式
            logger.info("开始评估...")
            val_metrics = trainer.validate()
            logger.info(f"验证结果: {val_metrics}")
            
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise e


if __name__ == '__main__':
    main()