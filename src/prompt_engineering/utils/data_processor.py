"""
数据处理工具模块
处理PENS数据集和用户数据
"""

import pandas as pd
import pickle
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import re

from ..core.config import config_manager

logger = logging.getLogger(__name__)


class PENSDataLoader:
    """PENS数据集加载器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            # 获取项目根目录下的data文件夹
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            data_dir = project_root / "data" / "raw"
        
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        
    def load_data(self, split: str = "all") -> Dict[str, Any]:
        """
        加载数据
        
        Args:
            split: 数据集分割，可选 "train", "valid", "test", "all"
            
        Returns:
            加载的数据字典
        """
        data = {}
        
        if split in ["train", "all"]:
            data["train"] = self._load_split("train")
        
        if split in ["valid", "all"]:
            data["valid"] = self._load_split("valid")
            
        if split in ["test", "all"]:
            data["test"] = self._load_split("test")
        
        return data
    
    def _load_split(self, split: str) -> List[Dict[str, Any]]:
        """
        加载特定分割的数据
        
        Args:
            split: 数据分割名称
            
        Returns:
            数据列表
        """
        # 尝试加载pickle文件（更快）
        pickle_file = self.data_dir / f"{split}.pkl"
        if pickle_file.exists():
            logger.info(f"加载pickle文件: {pickle_file}")
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        
        # 如果pickle文件不存在，尝试加载TSV文件
        tsv_file = self.data_dir / f"{split}.tsv"
        if tsv_file.exists():
            logger.info(f"加载TSV文件: {tsv_file}")
            return self._load_tsv(tsv_file)
        
        # 特殊处理测试文件
        if split == "test":
            test_file = self.data_dir / "personalized_test.tsv"
            if test_file.exists():
                logger.info(f"加载个性化测试文件: {test_file}")
                return self._load_tsv(test_file)
        
        logger.warning(f"未找到 {split} 数据文件")
        return []
    
    def _load_tsv(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        加载TSV文件
        
        Args:
            file_path: TSV文件路径
            
        Returns:
            数据列表
        """
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
            logger.info(f"成功加载 {len(df)} 条数据")
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"加载TSV文件失败: {e}")
            return []
    
    def get_user_history(self, user_id: str, data: List[Dict[str, Any]]) -> List[str]:
        """
        获取用户历史点击标题
        
        Args:
            user_id: 用户ID
            data: 数据列表
            
        Returns:
            用户历史标题列表
        """
        user_data = [item for item in data if item.get('user_id') == user_id]
        
        # 按时间排序（如果有时间戳）
        if user_data and 'timestamp' in user_data[0]:
            user_data.sort(key=lambda x: x.get('timestamp', 0))
        
        # 提取标题
        history_titles = []
        for item in user_data:
            title = item.get('title') or item.get('original_title')
            if title:
                history_titles.append(title)
        
        return history_titles
    
    def get_sample_for_generation(self, data: List[Dict[str, Any]], index: int = 0) -> Dict[str, Any]:
        """
        获取用于生成的样本
        
        Args:
            data: 数据列表
            index: 样本索引
            
        Returns:
            样本字典，包含user_id, news_content, original_title等
        """
        if 0 <= index < len(data):
            return data[index]
        return {}


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self):
        self.config = config_manager.get_data_config()
    
    def preprocess_news_content(self, content: str) -> str:
        """
        预处理新闻内容
        
        Args:
            content: 原始新闻内容
            
        Returns:
            预处理后的内容
        """
        if not content:
            return ""
        
        # 去除多余空白字符
        content = re.sub(r'\s+', ' ', content.strip())
        
        # 限制长度
        max_length = self.config.get("max_content_length", 1000)
        if len(content) > max_length:
            content = content[:max_length] + "..."
        
        return content
    
    def preprocess_title(self, title: str) -> str:
        """
        预处理标题
        
        Args:
            title: 原始标题
            
        Returns:
            预处理后的标题
        """
        if not title:
            return ""
        
        # 去除多余空白字符
        title = re.sub(r'\s+', ' ', title.strip())
        
        # 限制长度
        max_length = self.config.get("max_title_length", 50)
        if len(title) > max_length:
            title = title[:max_length]
        
        return title
    
    def filter_user_history(self, history: List[str]) -> List[str]:
        """
        过滤用户历史标题
        
        Args:
            history: 原始历史标题列表
            
        Returns:
            过滤后的历史标题列表
        """
        # 预处理每个标题
        processed_history = []
        for title in history:
            processed_title = self.preprocess_title(title)
            if processed_title:  # 只保留非空标题
                processed_history.append(processed_title)
        
        # 限制历史数量
        max_history = self.config.get("max_history_titles", 20)
        if len(processed_history) > max_history:
            processed_history = processed_history[-max_history:]  # 保留最近的
        
        return processed_history
    
    def prepare_generation_input(
        self, 
        sample: Dict[str, Any], 
        user_history: List[str]
    ) -> Tuple[str, List[str]]:
        """
        准备生成输入
        
        Args:
            sample: 样本数据
            user_history: 用户历史
            
        Returns:
            (新闻内容, 过滤后的用户历史)
        """
        # 提取新闻内容
        news_content = sample.get('content') or sample.get('body') or sample.get('text', '')
        news_content = self.preprocess_news_content(news_content)
        
        # 过滤用户历史
        filtered_history = self.filter_user_history(user_history)
        
        return news_content, filtered_history


class DataSampler:
    """数据采样器"""
    
    def __init__(self, data_loader: PENSDataLoader):
        self.data_loader = data_loader
        self.preprocessor = DataPreprocessor()
    
    def sample_for_experiment(
        self, 
        split: str = "test", 
        num_samples: int = 100,
        min_history_length: int = 5
    ) -> List[Dict[str, Any]]:
        """
        为实验采样数据
        
        Args:
            split: 数据分割
            num_samples: 采样数量
            min_history_length: 最小历史长度
            
        Returns:
            采样的数据列表
        """
        data = self.data_loader.load_data(split)[split]
        
        # 获取有足够历史的样本
        valid_samples = []
        processed_users = set()
        
        for sample in data:
            user_id = sample.get('user_id')
            if user_id in processed_users:
                continue
            
            # 获取用户历史
            user_history = self.data_loader.get_user_history(user_id, data)
            
            if len(user_history) >= min_history_length:
                news_content, filtered_history = self.preprocessor.prepare_generation_input(
                    sample, user_history
                )
                
                if news_content and filtered_history:
                    valid_samples.append({
                        'user_id': user_id,
                        'news_content': news_content,
                        'user_history': filtered_history,
                        'original_title': sample.get('title') or sample.get('original_title', ''),
                        'sample_data': sample
                    })
                    processed_users.add(user_id)
                    
                    if len(valid_samples) >= num_samples:
                        break
        
        logger.info(f"采样得到 {len(valid_samples)} 个有效样本")
        return valid_samples
    
    def get_evaluation_samples(self, num_samples: int = 50) -> List[Dict[str, Any]]:
        """
        获取评估样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            评估样本列表
        """
        return self.sample_for_experiment("test", num_samples, min_history_length=3)


def test_data_loader():
    """测试数据加载器"""
    try:
        # 初始化数据加载器
        loader = PENSDataLoader()
        
        # 加载测试数据
        logger.info("加载测试数据...")
        data = loader.load_data("test")
        
        if "test" in data and data["test"]:
            test_data = data["test"]
            logger.info(f"测试数据样本数: {len(test_data)}")
            
            # 显示第一个样本
            if test_data:
                sample = test_data[0]
                logger.info(f"样本键: {list(sample.keys())}")
                logger.info(f"第一个样本: {sample}")
            
            # 测试数据采样
            sampler = DataSampler(loader)
            samples = sampler.sample_for_experiment("test", num_samples=5)
            logger.info(f"采样得到 {len(samples)} 个样本")
            
            if samples:
                logger.info(f"第一个采样样本: {samples[0]}")
            
            return True
        else:
            logger.warning("未找到测试数据")
            return False
            
    except Exception as e:
        logger.error(f"数据加载测试失败: {e}")
        return False


if __name__ == "__main__":
    # 设置日志
    config_manager.setup_logging()
    
    # 运行测试
    test_data_loader()