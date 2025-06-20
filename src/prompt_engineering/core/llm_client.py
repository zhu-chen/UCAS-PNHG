"""
LLM API客户端模块
支持多个API提供商，包括SiliconFlow、OpenAI、智谱AI等
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from abc import ABC, abstractmethod
import time

from .config import config_manager

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, provider_config: Dict[str, Any]):
        self.config = provider_config
        self.base_url = provider_config.get("base_url")
        self.api_key = provider_config.get("api_key")
        self.models = provider_config.get("models", [])
        
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """聊天完成接口"""
        pass
    
    @abstractmethod
    async def chat_completion_stream(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式聊天完成接口"""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI兼容的API客户端（支持SiliconFlow、OpenAI等）"""
    
    def __init__(self, provider_config: Dict[str, Any]):
        super().__init__(provider_config)
        self.session = None
    
    async def _get_session(self):
        """获取aiohttp会话"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=config_manager.get_config("experiment.timeout", 30))
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        聊天完成接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            API响应
        """
        session = await self._get_session()
        
        # 构建请求数据
        data = {
            "model": model or self.models[0] if self.models else "gpt-3.5-turbo",
            "messages": messages,
            "temperature": kwargs.get("temperature", config_manager.get_config("model.temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", config_manager.get_config("model.max_tokens", 512)),
            "top_p": kwargs.get("top_p", config_manager.get_config("model.top_p", 0.9)),
        }
        
        # 添加其他参数
        for key, value in kwargs.items():
            if key not in ["temperature", "max_tokens", "top_p"]:
                data[key] = value
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/chat/completions"
        
        max_retries = config_manager.get_config("experiment.max_retries", 3)
        
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"API请求失败 (状态码: {response.status}): {error_text}")
                        if attempt == max_retries - 1:
                            raise Exception(f"API请求失败: {error_text}")
                        
            except Exception as e:
                logger.error(f"API请求异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    async def chat_completion_stream(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式聊天完成接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Yields:
            流式响应数据
        """
        session = await self._get_session()
        
        # 构建请求数据
        data = {
            "model": model or self.models[0] if self.models else "gpt-3.5-turbo",
            "messages": messages,
            "temperature": kwargs.get("temperature", config_manager.get_config("model.temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", config_manager.get_config("model.max_tokens", 512)),
            "top_p": kwargs.get("top_p", config_manager.get_config("model.top_p", 0.9)),
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/chat/completions"
        
        async with session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"流式API请求失败: {error_text}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue


class ZhipuClient(BaseLLMClient):
    """智谱AI客户端"""
    
    def __init__(self, provider_config: Dict[str, Any]):
        super().__init__(provider_config)
        self.session = None
    
    async def _get_session(self):
        """获取aiohttp会话"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=config_manager.get_config("experiment.timeout", 30))
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """关闭会话"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """智谱AI聊天完成接口"""
        # 智谱AI使用类似OpenAI的接口，但可能有细微差异
        session = await self._get_session()
        
        data = {
            "model": model or self.models[0] if self.models else "glm-4",
            "messages": messages,
            "temperature": kwargs.get("temperature", config_manager.get_config("model.temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", config_manager.get_config("model.max_tokens", 512)),
            "top_p": kwargs.get("top_p", config_manager.get_config("model.top_p", 0.9)),
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/chat/completions"
        
        max_retries = config_manager.get_config("experiment.max_retries", 3)
        
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"智谱AI请求失败 (状态码: {response.status}): {error_text}")
                        if attempt == max_retries - 1:
                            raise Exception(f"智谱AI请求失败: {error_text}")
                        
            except Exception as e:
                logger.error(f"智谱AI请求异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    
    async def chat_completion_stream(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """智谱AI流式聊天完成接口"""
        # 实现流式接口（如果智谱AI支持）
        session = await self._get_session()
        
        data = {
            "model": model or self.models[0] if self.models else "glm-4",
            "messages": messages,
            "temperature": kwargs.get("temperature", config_manager.get_config("model.temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", config_manager.get_config("model.max_tokens", 512)),
            "top_p": kwargs.get("top_p", config_manager.get_config("model.top_p", 0.9)),
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        url = f"{self.base_url}/chat/completions"
        
        async with session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"智谱AI流式请求失败: {error_text}")
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        yield data
                    except json.JSONDecodeError:
                        continue


class LLMClientFactory:
    """LLM客户端工厂"""
    
    _clients = {}
    
    @classmethod
    def get_client(cls, provider: str) -> BaseLLMClient:
        """
        获取指定提供商的客户端
        
        Args:
            provider: 提供商名称
            
        Returns:
            LLM客户端实例
        """
        if provider not in cls._clients:
            provider_config = config_manager.get_provider_config(provider)
            
            if not provider_config.get("api_key"):
                raise ValueError(f"未找到 {provider} 的API密钥，请检查私有配置文件")
            
            if provider == "zhipu":
                cls._clients[provider] = ZhipuClient(provider_config)
            else:
                # 对于SiliconFlow、OpenAI等使用OpenAI兼容客户端
                cls._clients[provider] = OpenAICompatibleClient(provider_config)
        
        return cls._clients[provider]
    
    @classmethod
    async def close_all(cls):
        """关闭所有客户端"""
        for client in cls._clients.values():
            await client.close()
        cls._clients.clear()


async def test_llm_client():
    """测试LLM客户端"""
    try:
        # 获取主要提供商
        primary_provider = config_manager.get_config("model.primary_provider", "siliconflow")
        client = LLMClientFactory.get_client(primary_provider)
        
        # 测试消息
        messages = [
            {"role": "system", "content": "你是一个专业的新闻标题生成助手。"},
            {"role": "user", "content": "请为这条新闻生成一个吸引人的标题：人工智能在医疗领域取得了重大突破。"}
        ]
        
        logger.info(f"测试 {primary_provider} API...")
        response = await client.chat_completion(messages)
        
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            logger.info(f"API测试成功，响应: {content}")
            return True
        else:
            logger.error(f"API响应格式异常: {response}")
            return False
            
    except Exception as e:
        logger.error(f"API测试失败: {str(e)}")
        return False
    finally:
        await LLMClientFactory.close_all()


if __name__ == "__main__":
    # 设置日志
    config_manager.setup_logging()
    
    # 运行测试
    asyncio.run(test_llm_client())