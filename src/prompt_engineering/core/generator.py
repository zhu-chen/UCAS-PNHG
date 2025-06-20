"""
个性化标题生成器
使用提示工程和LLM API生成个性化新闻标题
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

from ..core.config import config_manager
from ..core.llm_client import LLMClientFactory
from ..prompts.templates import template_manager
from ..utils.data_processor import DataPreprocessor

logger = logging.getLogger(__name__)


class PersonalizedTitleGenerator:
    """个性化标题生成器"""
    
    def __init__(self, provider: str = None, model: str = None):
        """
        初始化生成器
        
        Args:
            provider: API提供商，默认使用配置中的主要提供商
            model: 模型名称，默认使用配置中的默认模型
        """
        self.provider = provider or config_manager.get_config("model.primary_provider", "siliconflow")
        self.model = model or config_manager.get_config("model.default_model")
        self.client = None
        self.preprocessor = DataPreprocessor()
        
        # 生成统计
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
    
    async def _get_client(self):
        """获取LLM客户端"""
        if self.client is None:
            self.client = LLMClientFactory.get_client(self.provider)
        return self.client
    
    async def generate_title(
        self, 
        news_content: str, 
        user_history: List[str], 
        strategy: str = "basic",
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成个性化标题
        
        Args:
            news_content: 新闻内容
            user_history: 用户历史标题列表
            strategy: 提示策略
            **kwargs: 其他参数
            
        Returns:
            生成结果字典
        """
        start_time = time.time()
        self.generation_stats["total_requests"] += 1
        
        try:
            # 获取提示模板
            template = template_manager.get_template(strategy)
            if not template:
                raise ValueError(f"未找到策略 '{strategy}' 的模板")
            
            # 预处理输入
            processed_content = self.preprocessor.preprocess_news_content(news_content)
            processed_history = self.preprocessor.filter_user_history(user_history)
            
            if not processed_content:
                raise ValueError("新闻内容为空")
            
            if not processed_history:
                logger.warning("用户历史为空，可能影响个性化效果")
            
            # 格式化提示
            messages = template.format_prompt(
                processed_content, 
                processed_history, 
                **kwargs
            )
            
            # 调用LLM API
            client = await self._get_client()
            response = await client.chat_completion(
                messages=messages,
                model=self.model,
                **kwargs
            )
            
            # 提取生成的标题
            generated_title = self._extract_title_from_response(response)
            
            # 后处理标题
            final_title = self.preprocessor.preprocess_title(generated_title)
            
            if not final_title:
                raise ValueError("生成的标题为空")
            
            # 计算生成时间
            generation_time = time.time() - start_time
            self.generation_stats["successful_generations"] += 1
            self.generation_stats["total_time"] += generation_time
            self.generation_stats["average_time"] = (
                self.generation_stats["total_time"] / self.generation_stats["total_requests"]
            )
            
            result = {
                "success": True,
                "generated_title": final_title,
                "strategy": strategy,
                "provider": self.provider,
                "model": self.model,
                "generation_time": generation_time,
                "input_length": len(processed_content),
                "history_length": len(processed_history),
                "raw_response": response,
                "processed_input": {
                    "news_content": processed_content,
                    "user_history": processed_history
                }
            }
            
            logger.info(f"成功生成标题: {final_title} (用时: {generation_time:.2f}s)")
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            self.generation_stats["failed_generations"] += 1
            
            error_result = {
                "success": False,
                "error": str(e),
                "strategy": strategy,
                "provider": self.provider,
                "model": self.model,
                "generation_time": generation_time
            }
            
            logger.error(f"标题生成失败: {e} (用时: {generation_time:.2f}s)")
            return error_result
    
    def _extract_title_from_response(self, response: Dict[str, Any]) -> str:
        """
        从API响应中提取标题
        
        Args:
            response: API响应
            
        Returns:
            提取的标题
        """
        try:
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                
                # 对于思维链策略，查找"3. 个性化标题："后的内容
                if "3. 个性化标题：" in content:
                    # 找到"3. 个性化标题："的位置
                    title_start = content.find("3. 个性化标题：")
                    if title_start != -1:
                        # 提取该行的内容
                        title_line = content[title_start:].split('\n')[0]
                        # 移除前缀"3. 个性化标题："
                        title = title_line.replace("3. 个性化标题：", "").strip()
                        if title and len(title) <= 100:
                            return title
                
                # 通用标题提取逻辑 - 查找常见的标题标记
                title_patterns = [
                    "标题：",
                    "生成标题：", 
                    "个性化标题：",
                    "最终标题："
                ]
                
                for pattern in title_patterns:
                    if pattern in content:
                        # 找到模式后提取标题
                        pattern_start = content.find(pattern)
                        title_line = content[pattern_start:].split('\n')[0]
                        title = title_line.replace(pattern, "").strip()
                        if title and 5 <= len(title) <= 100:
                            return title
                
                # 如果没有找到明确标记，尝试提取合理的单行内容
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                # 优先选择中等长度的行作为标题
                for line in lines:
                    # 排除明显的分析文字和格式标记
                    if (10 <= len(line) <= 80 and 
                        not any(word in line for word in ["分析", "要点", "偏好", "步骤", "##", "**", "根据", "基于"]) and
                        not line.endswith("：") and
                        not line.startswith("1.") and
                        not line.startswith("2.") and
                        not line.startswith("3.")):
                        return line
                
                # 最后的备选：返回第一行（如果合理）
                if lines and 5 <= len(lines[0]) <= 100:
                    return lines[0]
                
                # 如果所有方法都失败，返回清理后的内容（截断）
                cleaned_content = content.strip()
                if len(cleaned_content) > 100:
                    cleaned_content = cleaned_content[:100]
                
                return cleaned_content
            else:
                raise ValueError("API响应格式异常")
                
        except Exception as e:
            logger.error(f"提取标题失败: {e}")
            raise
    
    async def generate_batch(
        self, 
        samples: List[Dict[str, Any]], 
        strategy: str = "basic",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量生成标题
        
        Args:
            samples: 样本列表，每个样本包含news_content和user_history
            strategy: 提示策略
            **kwargs: 其他参数
            
        Returns:
            生成结果列表
        """
        results = []
        batch_size = config_manager.get_config("experiment.batch_size", 10)
        max_concurrent = config_manager.get_config("experiment.max_concurrent_requests", 5)
        
        logger.info(f"开始批量生成，样本数: {len(samples)}, 策略: {strategy}")
        
        # 分批处理
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]
            
            # 创建并发任务
            tasks = []
            semaphore = asyncio.Semaphore(max_concurrent)
            
            for sample in batch_samples:
                task = self._generate_with_semaphore(
                    semaphore,
                    sample["news_content"],
                    sample["user_history"],
                    strategy,
                    sample,
                    **kwargs
                )
                tasks.append(task)
            
            # 等待当前批次完成
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"样本 {i+j} 生成异常: {result}")
                    results.append({
                        "success": False,
                        "error": str(result),
                        "sample_index": i + j
                    })
                else:
                    result["sample_index"] = i + j
                    results.append(result)
            
            logger.info(f"完成批次 {i//batch_size + 1}/{(len(samples)-1)//batch_size + 1}")
        
        # 统计结果
        successful = sum(1 for r in results if r.get("success"))
        logger.info(f"批量生成完成，成功: {successful}/{len(samples)}")
        
        return results
    
    async def _generate_with_semaphore(
        self, 
        semaphore: asyncio.Semaphore, 
        news_content: str, 
        user_history: List[str], 
        strategy: str,
        original_sample: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        带信号量控制的生成方法
        
        Args:
            semaphore: 信号量
            news_content: 新闻内容
            user_history: 用户历史
            strategy: 策略
            original_sample: 原始样本数据
            **kwargs: 其他参数
            
        Returns:
            生成结果
        """
        async with semaphore:
            result = await self.generate_title(
                news_content, 
                user_history, 
                strategy, 
                **kwargs
            )
            
            # 添加原始样本信息
            result["original_sample"] = original_sample
            return result
    
    async def compare_strategies(
        self, 
        news_content: str, 
        user_history: List[str],
        strategies: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        比较不同策略的生成效果
        
        Args:
            news_content: 新闻内容
            user_history: 用户历史
            strategies: 策略列表，默认比较所有可用策略
            
        Returns:
            策略比较结果字典
        """
        if strategies is None:
            strategies = template_manager.list_strategies()
        
        logger.info(f"比较策略: {strategies}")
        
        results = {}
        for strategy in strategies:
            try:
                result = await self.generate_title(news_content, user_history, strategy)
                results[strategy] = result
            except Exception as e:
                logger.error(f"策略 {strategy} 生成失败: {e}")
                results[strategy] = {
                    "success": False,
                    "error": str(e),
                    "strategy": strategy
                }
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        return self.generation_stats.copy()
    
    def reset_statistics(self):
        """重置统计信息"""
        self.generation_stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_time": 0.0,
            "average_time": 0.0
        }
    
    async def close(self):
        """关闭生成器"""
        if self.client:
            await LLMClientFactory.close_all()


async def test_generator():
    """测试生成器"""
    try:
        # 初始化生成器
        generator = PersonalizedTitleGenerator()
        
        # 测试数据
        news_content = """
        据最新研究显示，人工智能技术在医疗诊断领域取得了重大突破。
        研究团队开发的AI系统能够在几秒钟内准确诊断多种疾病，准确率达到95%以上。
        这项技术有望大大提高医疗效率，缓解医生短缺问题。
        """
        
        user_history = [
            "AI技术发展迅速，未来前景广阔",
            "医疗行业数字化转型加速",
            "人工智能助力精准医疗",
            "科技创新推动医疗进步"
        ]
        
        # 测试基础策略
        logger.info("测试基础策略...")
        result = await generator.generate_title(news_content, user_history, "basic")
        
        if result["success"]:
            logger.info(f"生成成功: {result['generated_title']}")
            
            # 测试策略比较
            logger.info("测试策略比较...")
            comparison = await generator.compare_strategies(news_content, user_history)
            
            for strategy, result in comparison.items():
                if result["success"]:
                    logger.info(f"{strategy}: {result['generated_title']}")
                else:
                    logger.error(f"{strategy}: 失败 - {result.get('error')}")
            
            # 显示统计信息
            stats = generator.get_statistics()
            logger.info(f"生成统计: {stats}")
            
            return True
        else:
            logger.error(f"生成失败: {result.get('error')}")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False
    finally:
        await generator.close()


if __name__ == "__main__":
    # 设置日志
    config_manager.setup_logging()
    
    # 运行测试
    asyncio.run(test_generator())