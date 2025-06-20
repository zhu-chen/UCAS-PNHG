"""
提示工程模块
基于LLM API的个性化新闻标题生成
"""

from .core.config import config_manager
from .core.generator import PersonalizedTitleGenerator
from .core.llm_client import LLMClientFactory
from .prompts.templates import template_manager
from .utils.data_processor import PENSDataLoader, DataSampler
from .experiments.runner import ExperimentRunner

__version__ = "1.0.0"
__author__ = "AI Research Team"

# 设置日志
config_manager.setup_logging()

__all__ = [
    "config_manager",
    "PersonalizedTitleGenerator", 
    "LLMClientFactory",
    "template_manager",
    "PENSDataLoader",
    "DataSampler",
    "ExperimentRunner"
]