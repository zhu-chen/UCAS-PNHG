"""
日志工具
设置和管理训练日志
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    设置日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径
        console_output: 是否输出到控制台
    
    Returns:
        配置好的日志器
    """
    # 获取或创建日志器
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    
    # 设置日志级别
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 避免重复日志
    logger.propagate = False
    
    return logger


def get_training_logger(log_dir: str = "./logs") -> logging.Logger:
    """
    获取训练专用日志器
    
    Args:
        log_dir: 日志目录
    
    Returns:
        训练日志器
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"training_{timestamp}.log"
    
    return setup_logger(
        name="training",
        level=logging.INFO,
        log_file=str(log_file),
        console_output=True
    )


class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "./logs", experiment_name: str = "pens_baseline"):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建实验专用日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        # 设置日志器
        self.logger = setup_logger(
            name=f"{experiment_name}_logger",
            level=logging.INFO,
            log_file=str(self.log_file),
            console_output=True
        )
        
        self.logger.info(f"训练日志器初始化完成，日志文件: {self.log_file}")
    
    def log_config(self, config: dict):
        """记录配置信息"""
        self.logger.info("=" * 50)
        self.logger.info("实验配置:")
        self._log_dict(config, indent=2)
        self.logger.info("=" * 50)
    
    def log_model_info(self, model, total_params: int, trainable_params: int):
        """记录模型信息"""
        self.logger.info("=" * 50)
        self.logger.info("模型信息:")
        self.logger.info(f"  模型类型: {type(model).__name__}")
        self.logger.info(f"  总参数数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  参数利用率: {trainable_params/total_params*100:.1f}%")
        self.logger.info("=" * 50)
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录训练轮次开始"""
        self.logger.info(f"开始第 {epoch+1}/{total_epochs} 轮训练")
    
    def log_epoch_end(self, epoch: int, train_loss: float, valid_loss: float, metrics: dict):
        """记录训练轮次结束"""
        self.logger.info(f"第 {epoch+1} 轮训练完成:")
        self.logger.info(f"  训练损失: {train_loss:.4f}")
        self.logger.info(f"  验证损失: {valid_loss:.4f}")
        
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    def log_training_complete(self, best_epoch: int, best_score: float):
        """记录训练完成"""
        self.logger.info("=" * 50)
        self.logger.info("训练完成!")
        self.logger.info(f"  最佳轮次: {best_epoch+1}")
        self.logger.info(f"  最佳分数: {best_score:.4f}")
        self.logger.info("=" * 50)
    
    def log_test_results(self, test_metrics: dict):
        """记录测试结果"""
        self.logger.info("=" * 50)
        self.logger.info("测试结果:")
        self._log_dict(test_metrics, indent=2)
        self.logger.info("=" * 50)
    
    def log_error(self, error_msg: str, exception: Exception = None):
        """记录错误"""
        self.logger.error(f"错误: {error_msg}")
        if exception:
            self.logger.exception(f"异常详情: {str(exception)}")
    
    def _log_dict(self, data: dict, indent: int = 0):
        """递归记录字典"""
        for key, value in data.items():
            if isinstance(value, dict):
                self.logger.info(" " * indent + f"{key}:")
                self._log_dict(value, indent + 2)
            else:
                self.logger.info(" " * indent + f"{key}: {value}")


def log_system_info():
    """记录系统信息"""
    import torch
    import platform
    
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("系统信息:")
    logger.info(f"  Python版本: {platform.python_version()}")
    logger.info(f"  PyTorch版本: {torch.__version__}")
    logger.info(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA版本: {torch.version.cuda}")
        logger.info(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory}GB)")
    
    logger.info(f"  操作系统: {platform.system()} {platform.release()}")
    logger.info("=" * 50)