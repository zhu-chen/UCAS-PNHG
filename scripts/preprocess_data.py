#!/usr/bin/env python3
"""
预处理PENS数据集 - 全量数据版本
"""

import os
import sys
from pathlib import Path
import yaml
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baseline.data.preprocessor import PENSPreprocessor

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocessing.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 加载配置
    with open('configs/baseline/data_configs.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("开始预处理PENS数据集（全量数据）")
    
    # 创建预处理器
    preprocessor = PENSPreprocessor(config['preprocessing'])
    
    # 预处理数据
    try:
        train_path, valid_path, test_path = preprocessor.preprocess_raw_data(
            raw_data_dir='data/raw',
            output_dir='data/processed'
        )
        
        logger.info(f"预处理完成！")
        logger.info(f"训练集: {train_path}")
        logger.info(f"验证集: {valid_path}")
        logger.info(f"测试集: {test_path}")
        
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        raise e

if __name__ == '__main__':
    main()