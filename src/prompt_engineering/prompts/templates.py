"""
提示模板管理器
定义和管理各种提示工程策略的模板
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import random

class BasePromptTemplate(ABC):
    """提示模板基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def format_prompt(
        self, 
        news_content: str, 
        user_history: List[str], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        格式化提示
        
        Args:
            news_content: 新闻正文
            user_history: 用户历史点击标题列表
            **kwargs: 其他参数
            
        Returns:
            格式化后的消息列表
        """
        pass


class BasicPromptTemplate(BasePromptTemplate):
    """基础提示模板"""
    
    def __init__(self):
        super().__init__(
            name="基础个性化提示",
            description="基于用户历史标题直接生成个性化标题"
        )
    
    def format_prompt(
        self, 
        news_content: str, 
        user_history: List[str], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """格式化基础提示"""
        
        # 限制历史标题数量
        max_history = kwargs.get("max_history", 10)
        history_titles = user_history[:max_history] if len(user_history) > max_history else user_history
        
        # 构建用户历史信息
        history_text = "\n".join([f"- {title}" for title in history_titles])
        
        system_prompt = """你是一个专业的个性化新闻标题生成助手。你的任务是根据用户的历史阅读偏好，为给定的新闻内容生成一个吸引该用户的个性化标题。

要求：
1. 标题要准确反映新闻内容的核心信息
2. 标题要符合用户的阅读偏好和兴趣点
3. 标题要简洁有力，具有吸引力
4. 标题长度控制在50字以内
5. 只输出标题文本，不要其他内容"""

        user_prompt = f"""用户历史点击标题：
{history_text}

新闻内容：
{news_content}

请基于用户的历史阅读偏好，为这条新闻生成一个个性化的标题："""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class ChainOfThoughtPromptTemplate(BasePromptTemplate):
    """思维链提示模板"""
    
    def __init__(self):
        super().__init__(
            name="思维链推理",
            description="通过逐步分析用户偏好生成标题"
        )
    
    def format_prompt(
        self, 
        news_content: str, 
        user_history: List[str], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """格式化思维链提示"""
        
        max_history = kwargs.get("max_history", 15)
        history_titles = user_history[:max_history] if len(user_history) > max_history else user_history
        history_text = "\n".join([f"{i+1}. {title}" for i, title in enumerate(history_titles)])
        
        system_prompt = """你是一个专业的个性化新闻标题生成助手。请按照以下步骤进行思考和分析：

1. 首先分析用户的历史阅读偏好
2. 然后分析新闻内容的核心要点
3. 最后结合用户偏好生成个性化标题

请按照这个思维过程逐步分析，最后输出标题。"""

        user_prompt = f"""用户历史点击标题：
{history_text}

新闻内容：
{news_content}

请按照以下步骤分析：

步骤1：分析用户偏好
请分析用户历史点击标题，总结用户的阅读偏好，包括：
- 感兴趣的主题类型
- 偏好的标题风格
- 关注的关键词类型

步骤2：分析新闻内容
请分析新闻内容的核心要点：
- 主要事件或话题
- 关键信息点
- 新闻价值

步骤3：生成个性化标题
基于用户偏好和新闻内容，生成一个个性化标题（50字以内）："""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class RolePlayingPromptTemplate(BasePromptTemplate):
    """角色扮演提示模板"""
    
    def __init__(self):
        super().__init__(
            name="角色扮演",
            description="让模型扮演新闻编辑角色"
        )
    
    def format_prompt(
        self, 
        news_content: str, 
        user_history: List[str], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """格式化角色扮演提示"""
        
        max_history = kwargs.get("max_history", 12)
        history_titles = user_history[:max_history] if len(user_history) > max_history else user_history
        history_text = "\n".join([f"• {title}" for title in history_titles])
        
        system_prompt = """你是一位资深的新闻编辑，拥有20年的新闻标题创作经验。你精通用户画像分析和个性化内容推荐。

作为专业编辑，你需要：
1. 深入理解不同用户的阅读偏好
2. 准确把握新闻内容的核心价值
3. 创作出既准确又吸引人的个性化标题
4. 确保标题符合新闻伦理和专业标准

你的目标是为特定用户量身定制最合适的新闻标题。"""

        user_prompt = f"""现在有一位用户，他的历史阅读记录如下：

{history_text}

请你作为资深新闻编辑，为以下新闻内容创作一个最适合这位用户的标题：

新闻内容：
{news_content}

请考虑用户的阅读偏好，创作一个准确、吸引人且个性化的标题（50字以内）："""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


class FewShotPromptTemplate(BasePromptTemplate):
    """少样本学习提示模板"""
    
    def __init__(self):
        super().__init__(
            name="少样本学习",
            description="提供示例来指导生成"
        )
        
        # 预定义一些示例
        self.examples = [
            {
                "user_history": [
                    "科技巨头AI竞赛白热化，谁将成为最终赢家？",
                    "人工智能改变医疗行业，患者受益巨大",
                    "自动驾驶技术新突破，商用化进程加速"
                ],
                "news_content": "据最新报告显示，全球人工智能市场规模预计将在2025年达到1900亿美元，比2020年增长超过200%。报告指出，机器学习、自然语言处理和计算机视觉是推动增长的主要技术领域。",
                "generated_title": "AI市场爆发式增长！2025年将突破1900亿美元大关"
            },
            {
                "user_history": [
                    "股市震荡，投资者如何应对？",
                    "央行政策调整，对经济影响几何？",
                    "房地产市场回暖，购房时机到了吗？"
                ],
                "news_content": "国家统计局今日发布数据显示，10月份居民消费价格指数(CPI)同比上涨2.1%，环比上涨0.2%。食品价格有所回落，但服务价格继续上涨。",
                "generated_title": "CPI数据出炉：通胀压力温和，消费复苏信号明确"
            }
        ]
    
    def format_prompt(
        self, 
        news_content: str, 
        user_history: List[str], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """格式化少样本学习提示"""
        
        max_history = kwargs.get("max_history", 10)
        history_titles = user_history[:max_history] if len(user_history) > max_history else user_history
        history_text = "\n".join([f"- {title}" for title in history_titles])
        
        # 选择最相关的示例
        selected_examples = self._select_examples(user_history, kwargs.get("num_examples", 2))
        
        system_prompt = """你是一个专业的个性化新闻标题生成助手。请根据提供的示例，学习如何为不同用户生成个性化标题。

重要原则：
1. 分析用户的历史阅读偏好
2. 提取新闻内容的核心信息
3. 生成符合用户兴趣的个性化标题
4. 标题要准确、简洁、有吸引力"""

        # 构建示例文本
        examples_text = ""
        for i, example in enumerate(selected_examples, 1):
            example_history = "\n".join([f"  - {title}" for title in example["user_history"]])
            examples_text += f"""
示例{i}：
用户历史：
{example_history}
新闻内容：{example["news_content"]}
生成标题：{example["generated_title"]}
"""

        user_prompt = f"""请参考以下示例：{examples_text}

现在请为以下用户生成个性化标题：

用户历史点击标题：
{history_text}

新闻内容：
{news_content}

请生成一个个性化标题（50字以内）："""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _select_examples(self, user_history: List[str], num_examples: int) -> List[Dict]:
        """选择最相关的示例"""
        # 简单实现：随机选择示例
        # 实际应用中可以基于相似度选择
        selected = random.sample(self.examples, min(num_examples, len(self.examples)))
        return selected


class PromptTemplateManager:
    """提示模板管理器"""
    
    def __init__(self):
        self.templates = {
            "basic": BasicPromptTemplate(),
            "chain_of_thought": ChainOfThoughtPromptTemplate(),
            "role_playing": RolePlayingPromptTemplate(),
            "few_shot": FewShotPromptTemplate()
        }
    
    def get_template(self, strategy: str) -> Optional[BasePromptTemplate]:
        """
        获取指定策略的模板
        
        Args:
            strategy: 策略名称
            
        Returns:
            提示模板实例
        """
        return self.templates.get(strategy)
    
    def list_strategies(self) -> List[str]:
        """获取所有可用策略列表"""
        return list(self.templates.keys())
    
    def get_strategy_info(self, strategy: str) -> Dict[str, str]:
        """
        获取策略信息
        
        Args:
            strategy: 策略名称
            
        Returns:
            策略信息字典
        """
        template = self.templates.get(strategy)
        if template:
            return {
                "name": template.name,
                "description": template.description
            }
        return {}

# 全局模板管理器实例
template_manager = PromptTemplateManager()