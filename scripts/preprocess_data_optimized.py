#!/usr/bin/env python3
"""
预处理PENS数据集 - 内存优化版本
"""

import os
import sys
import gc
import psutil
from pathlib import Path
import yaml
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.baseline.data.preprocessor import PENSPreprocessor

def setup_logging():
    """设置日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/preprocessing.log'),
            logging.StreamHandler()
        ]
    )

def monitor_memory():
    """监控内存使用"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return memory_mb

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 监控初始内存
    initial_memory = monitor_memory()
    logger.info(f"初始内存使用: {initial_memory:.2f} MB")
    
    # 加载配置
    with open('configs/baseline/data_configs.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("开始预处理PENS数据集（分块处理版本）")
    
    # 创建预处理器
    preprocessor = PENSPreprocessor(config['preprocessing'])
    
    # 预处理数据
    try:
        train_path, valid_path, test_path = preprocessor.preprocess_raw_data(
            raw_data_dir='data/raw',
            output_dir='data/processed'
        )
        
        # 监控最终内存
        final_memory = monitor_memory()
        logger.info(f"最终内存使用: {final_memory:.2f} MB")
        logger.info(f"内存增长: {final_memory - initial_memory:.2f} MB")
        
        logger.info(f"预处理完成！")
        logger.info(f"训练集: {train_path}")
        logger.info(f"验证集: {valid_path}")
        logger.info(f"测试集: {test_path}")
        
        # 检查生成的文件大小
        for path in [train_path, valid_path, test_path]:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                logger.info(f"文件大小 {Path(path).name}: {size_mb:.2f} MB")
        
    except Exception as e:
        logger.error(f"预处理失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise e
    finally:
        # 清理内存
        gc.collect()

if __name__ == '__main__':
    main()