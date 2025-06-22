"""
检查点管理器
处理模型检查点的保存和加载
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最大保存检查点数量
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        
        # 检查点文件命名格式
        self.checkpoint_pattern = "checkpoint_epoch_{}.pth"
        self.best_checkpoint_name = "best_model.pth"
        
        logger.info(f"检查点管理器初始化，保存目录: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self, 
        checkpoint: Dict[str, Any], 
        epoch: Optional[int] = None,
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        保存检查点
        
        Args:
            checkpoint: 检查点数据
            epoch: 训练轮次
            is_best: 是否为最佳模型
            filename: 自定义文件名
        """
        # 添加保存时间戳
        checkpoint['save_time'] = datetime.now().isoformat()
        
        if filename:
            checkpoint_path = self.checkpoint_dir / filename
        elif is_best:
            checkpoint_path = self.checkpoint_dir / self.best_checkpoint_name
        else:
            if epoch is None:
                epoch = checkpoint.get('epoch', 0)
            checkpoint_path = self.checkpoint_dir / self.checkpoint_pattern.format(epoch)
        
        # 保存检查点
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 保存检查点信息
        if not is_best:
            self._save_checkpoint_info(checkpoint_path, checkpoint)
        
        # 清理旧检查点
        if not is_best and self.max_checkpoints > 0:
            self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        
        Returns:
            检查点数据
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"检查点文件不存在: {checkpoint_path}")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            logger.info(f"检查点加载成功: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"检查点加载失败: {e}")
            return None
    
    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载最佳模型检查点"""
        best_path = self.checkpoint_dir / self.best_checkpoint_name
        return self.load_checkpoint(str(best_path))
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """加载最新的检查点"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if not checkpoint_files:
            logger.info("未找到训练检查点")
            return None
        
        # 按epoch号排序，获取最新的
        latest_file = max(checkpoint_files, key=lambda x: self._extract_epoch_from_filename(x.name))
        return self.load_checkpoint(str(latest_file))
    
    def list_checkpoints(self) -> list:
        """列出所有检查点"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        checkpoint_info = []
        
        for file_path in checkpoint_files:
            epoch = self._extract_epoch_from_filename(file_path.name)
            info_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
            
            info = {
                'epoch': epoch,
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            # 加载额外信息
            if info_path.exists():
                try:
                    with open(info_path, 'r', encoding='utf-8') as f:
                        extra_info = json.load(f)
                        info.update(extra_info)
                except:
                    pass
            
            checkpoint_info.append(info)
        
        # 按epoch排序
        checkpoint_info.sort(key=lambda x: x['epoch'])
        return checkpoint_info
    
    def delete_checkpoint(self, epoch: int):
        """删除指定轮次的检查点"""
        checkpoint_path = self.checkpoint_dir / self.checkpoint_pattern.format(epoch)
        info_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"删除检查点: {checkpoint_path}")
        
        if info_path.exists():
            info_path.unlink()
    
    def _save_checkpoint_info(self, checkpoint_path: Path, checkpoint: Dict[str, Any]):
        """保存检查点额外信息"""
        epoch = checkpoint.get('epoch', 0)
        info_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
        
        info = {
            'epoch': epoch,
            'best_score': checkpoint.get('best_score', 0.0),
            'save_time': checkpoint.get('save_time'),
            'file_path': str(checkpoint_path),
            'file_size': checkpoint_path.stat().st_size
        }
        
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
        except Exception as e:
            logger.warning(f"保存检查点信息失败: {e}")
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # 按epoch号排序
        checkpoint_files.sort(key=lambda x: self._extract_epoch_from_filename(x.name))
        
        # 删除最旧的检查点
        files_to_delete = checkpoint_files[:-self.max_checkpoints]
        
        for file_path in files_to_delete:
            epoch = self._extract_epoch_from_filename(file_path.name)
            self.delete_checkpoint(epoch)
    
    def _extract_epoch_from_filename(self, filename: str) -> int:
        """从文件名提取epoch号"""
        try:
            # 提取 checkpoint_epoch_{epoch}.pth 中的 epoch
            parts = filename.split('_')
            epoch_part = parts[2].split('.')[0]  # 去掉.pth扩展名
            return int(epoch_part)
        except:
            return 0
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """获取检查点统计信息"""
        checkpoints = self.list_checkpoints()
        best_path = self.checkpoint_dir / self.best_checkpoint_name
        
        stats = {
            'total_checkpoints': len(checkpoints),
            'checkpoint_dir': str(self.checkpoint_dir),
            'max_checkpoints': self.max_checkpoints,
            'has_best_model': best_path.exists()
        }
        
        if checkpoints:
            stats['latest_epoch'] = max(cp['epoch'] for cp in checkpoints)
            stats['earliest_epoch'] = min(cp['epoch'] for cp in checkpoints)
            
            # 计算总大小
            total_size = sum(cp['file_size'] for cp in checkpoints)
            if best_path.exists():
                total_size += best_path.stat().st_size
            
            stats['total_size_mb'] = total_size / (1024 * 1024)
        
        return stats


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    best_score: Optional[float] = None,
    **kwargs
):
    """
    保存模型检查点的便捷函数
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        step: 当前步数
        loss: 当前损失
        path: 保存路径
        best_score: 最佳分数
        **kwargs: 其他要保存的数据
    """
    # 确保保存目录存在
    save_dir = Path(path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建检查点数据
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'best_score': best_score,
        'save_time': datetime.now().isoformat(),
        **kwargs
    }
    
    # 保存检查点
    torch.save(checkpoint, path)
    logger.info(f"检查点已保存到: {path}")


def load_checkpoint(path: str, model=None, optimizer=None, scheduler=None, device='cpu'):
    """
    加载模型检查点的便捷函数
    
    Args:
        path: 检查点路径
        model: 模型（可选）
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
    
    Returns:
        检查点数据字典
    """
    if not os.path.exists(path):
        logger.error(f"检查点文件不存在: {path}")
        return None
    
    try:
        # 加载检查点
        checkpoint = torch.load(path, map_location=device)
        
        # 恢复模型状态
        if model and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("模型状态已恢复")
        
        # 恢复优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("优化器状态已恢复")
        
        # 恢复调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("学习率调度器状态已恢复")
        
        logger.info(f"检查点加载成功: {path}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"检查点加载失败: {e}")
        return None