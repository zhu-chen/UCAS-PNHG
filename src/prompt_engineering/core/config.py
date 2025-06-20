"""
配置管理模块
负责加载和管理项目配置，包括主配置和私有配置
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = None):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的configs
        """
        if config_dir is None:
            # 获取项目根目录
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            config_dir = project_root / "configs"
        
        self.config_dir = Path(config_dir)
        self.main_config = None
        self.private_config = None
        
        # 加载配置
        self._load_configs()
    
    def _load_configs(self):
        """加载主配置和私有配置"""
        # 加载主配置
        main_config_path = self.config_dir / "prompt_engineering.yaml"
        if main_config_path.exists():
            with open(main_config_path, 'r', encoding='utf-8') as f:
                self.main_config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"主配置文件不存在: {main_config_path}")
        
        # 加载私有配置
        private_config_path = self.config_dir / "private" / "api_keys.private.yaml"
        if private_config_path.exists():
            with open(private_config_path, 'r', encoding='utf-8') as f:
                self.private_config = yaml.safe_load(f)
        else:
            logging.warning(f"私有配置文件不存在: {private_config_path}")
            logging.warning("请复制模板文件并填入API密钥")
            self.private_config = {"api_keys": {}, "private_settings": {}}
    
    def get_config(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置键路径，使用点号分隔，如 "model.temperature"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        config = self.main_config
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        
        return config
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        获取API密钥
        
        Args:
            provider: API提供商名称
            
        Returns:
            API密钥
        """
        if self.private_config and "api_keys" in self.private_config:
            return self.private_config["api_keys"].get(provider)
        return None
    
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        获取API提供商配置
        
        Args:
            provider: API提供商名称
            
        Returns:
            提供商配置字典
        """
        providers_config = self.get_config("providers", {})
        provider_config = providers_config.get(provider, {})
        
        # 添加API密钥
        api_key = self.get_api_key(provider)
        if api_key:
            provider_config["api_key"] = api_key
        
        return provider_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get_config("model", {})
    
    def get_prompt_strategy_config(self, strategy: str) -> Dict[str, Any]:
        """
        获取提示策略配置
        
        Args:
            strategy: 策略名称
            
        Returns:
            策略配置字典
        """
        strategies = self.get_config("prompt_strategies", {})
        return strategies.get(strategy, {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """获取实验配置"""
        return self.get_config("experiment", {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.get_config("evaluation", {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据处理配置"""
        return self.get_config("data", {})
    
    def setup_logging(self):
        """设置日志配置"""
        logging_config = self.get_config("logging", {})
        
        # 创建日志目录
        log_file = logging_config.get("file", "logs/prompt_engineering.log")
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志
        logging.basicConfig(
            level=getattr(logging, logging_config.get("level", "INFO")),
            format=logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

# 全局配置管理器实例
config_manager = ConfigManager()