"""
配置管理工具
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """加载配置文件"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"已加载配置文件: {config_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(other_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config


def load_baseline_configs() -> Dict[str, Config]:
    """加载基线方法的所有配置文件"""
    # 获取项目根目录
    current_dir = Path(__file__).parent.parent.parent.parent
    config_dir = current_dir / "configs" / "baseline"
    
    configs = {}
    
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = Config(str(config_file))
    
    return configs


def load_configs(config_dir: str) -> Dict[str, Config]:
    """加载指定目录的所有配置文件"""
    config_dir = Path(config_dir)
    configs = {}
    
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = Config(str(config_file))
    
    return configs


def merge_configs(*configs: Config) -> Config:
    """合并多个配置"""
    merged = Config()
    
    for config in configs:
        merged.update(config.to_dict())
    
    return merged