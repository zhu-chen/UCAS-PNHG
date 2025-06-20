"""
实验运行器
用于运行提示工程实验和评估
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

from ..core.config import config_manager
from ..core.generator import PersonalizedTitleGenerator
from ..utils.data_processor import PENSDataLoader, DataSampler
from ..prompts.templates import template_manager

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, experiment_name: str = None):
        """
        初始化实验运行器
        
        Args:
            experiment_name: 实验名称，用于保存结果
        """
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data_loader = PENSDataLoader()
        self.data_sampler = DataSampler(self.data_loader)
        self.results_dir = self._setup_results_dir()
        
        # 实验配置
        self.config = config_manager.get_experiment_config()
        
        # 实验结果
        self.results = {
            "experiment_info": {
                "name": self.experiment_name,
                "start_time": None,
                "end_time": None,
                "config": self.config,
                "strategies": [],
                "total_samples": 0,
                "successful_generations": 0
            },
            "generation_results": [],
            "strategy_comparison": {},
            "statistics": {}
        }
    
    def _setup_results_dir(self) -> Path:
        """设置结果目录"""
        # 获取项目根目录
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        results_dir = project_root / "results" / "prompt_engineering" / self.experiment_name
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    async def run_single_strategy_experiment(
        self, 
        strategy: str,
        num_samples: int = 50,
        provider: str = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        运行单一策略实验
        
        Args:
            strategy: 提示策略
            num_samples: 样本数量
            provider: API提供商
            model: 模型名称
            
        Returns:
            实验结果
        """
        logger.info(f"开始运行单一策略实验: {strategy}")
        
        # 获取实验样本
        samples = self.data_sampler.sample_for_experiment("test", num_samples)
        if not samples:
            raise ValueError("未能获取有效的实验样本")
        
        # 初始化生成器
        generator = PersonalizedTitleGenerator(provider, model)
        
        try:
            # 批量生成
            results = await generator.generate_batch(samples, strategy)
            
            # 统计结果
            successful = sum(1 for r in results if r.get("success"))
            stats = generator.get_statistics()
            
            experiment_result = {
                "strategy": strategy,
                "provider": generator.provider,
                "model": generator.model,
                "total_samples": len(samples),
                "successful_generations": successful,
                "success_rate": successful / len(samples) if samples else 0,
                "generation_stats": stats,
                "results": results
            }
            
            # 保存结果
            self._save_strategy_result(experiment_result)
            
            logger.info(f"策略 {strategy} 实验完成，成功率: {experiment_result['success_rate']:.2%}")
            return experiment_result
            
        finally:
            await generator.close()
    
    async def run_strategy_comparison_experiment(
        self, 
        strategies: List[str] = None,
        num_samples: int = 30,
        provider: str = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        运行策略比较实验
        
        Args:
            strategies: 策略列表，默认比较所有策略
            num_samples: 样本数量
            provider: API提供商
            model: 模型名称
            
        Returns:
            比较实验结果
        """
        if strategies is None:
            strategies = template_manager.list_strategies()
        
        logger.info(f"开始运行策略比较实验，策略: {strategies}")
        
        # 获取实验样本
        samples = self.data_sampler.sample_for_experiment("test", num_samples)
        if not samples:
            raise ValueError("未能获取有效的实验样本")
        
        # 为每个策略运行实验
        comparison_results = {}
        
        for strategy in strategies:
            logger.info(f"运行策略: {strategy}")
            try:
                result = await self.run_single_strategy_experiment(
                    strategy, num_samples, provider, model
                )
                comparison_results[strategy] = result
            except Exception as e:
                logger.error(f"策略 {strategy} 实验失败: {e}")
                comparison_results[strategy] = {
                    "strategy": strategy,
                    "error": str(e),
                    "success": False
                }
        
        # 生成比较报告
        comparison_report = self._generate_comparison_report(comparison_results)
        
        # 保存比较结果
        self._save_comparison_result(comparison_results, comparison_report)
        
        logger.info("策略比较实验完成")
        return {
            "comparison_results": comparison_results,
            "comparison_report": comparison_report
        }
    
    async def run_comprehensive_experiment(
        self, 
        strategies: List[str] = None,
        num_samples: int = 100,
        providers: List[str] = None
    ) -> Dict[str, Any]:
        """
        运行综合实验
        
        Args:
            strategies: 策略列表
            num_samples: 样本数量
            providers: API提供商列表
            
        Returns:
            综合实验结果
        """
        logger.info("开始运行综合实验")
        
        self.results["experiment_info"]["start_time"] = datetime.now().isoformat()
        
        if strategies is None:
            strategies = template_manager.list_strategies()
        
        if providers is None:
            providers = [config_manager.get_config("model.primary_provider", "siliconflow")]
        
        self.results["experiment_info"]["strategies"] = strategies
        self.results["experiment_info"]["total_samples"] = num_samples
        
        comprehensive_results = {}
        
        # 为每个提供商运行实验
        for provider in providers:
            logger.info(f"运行提供商: {provider}")
            
            try:
                provider_results = await self.run_strategy_comparison_experiment(
                    strategies, num_samples, provider
                )
                comprehensive_results[provider] = provider_results
                
            except Exception as e:
                logger.error(f"提供商 {provider} 实验失败: {e}")
                comprehensive_results[provider] = {
                    "error": str(e),
                    "success": False
                }
        
        self.results["experiment_info"]["end_time"] = datetime.now().isoformat()
        self.results["strategy_comparison"] = comprehensive_results
        
        # 生成综合报告
        comprehensive_report = self._generate_comprehensive_report(comprehensive_results)
        self.results["comprehensive_report"] = comprehensive_report
        
        # 保存完整结果
        self._save_comprehensive_result()
        
        logger.info("综合实验完成")
        return self.results
    
    def _generate_comparison_report(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成策略比较报告"""
        report = {
            "summary": {},
            "performance_ranking": [],
            "detailed_analysis": {}
        }
        
        # 提取成功的结果
        successful_results = {
            k: v for k, v in comparison_results.items() 
            if v.get("success", True) and "success_rate" in v
        }
        
        if not successful_results:
            return report
        
        # 性能排名
        ranking = sorted(
            successful_results.items(),
            key=lambda x: x[1]["success_rate"],
            reverse=True
        )
        
        report["performance_ranking"] = [
            {
                "strategy": strategy,
                "success_rate": data["success_rate"],
                "avg_time": data["generation_stats"].get("average_time", 0),
                "total_samples": data["total_samples"]
            }
            for strategy, data in ranking
        ]
        
        # 摘要统计
        success_rates = [data["success_rate"] for data in successful_results.values()]
        avg_times = [data["generation_stats"].get("average_time", 0) for data in successful_results.values()]
        
        report["summary"] = {
            "total_strategies": len(successful_results),
            "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
            "avg_generation_time": sum(avg_times) / len(avg_times) if avg_times else 0,
            "best_strategy": ranking[0][0] if ranking else None,
            "worst_strategy": ranking[-1][0] if ranking else None
        }
        
        return report
    
    def _generate_comprehensive_report(self, comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            "overview": {},
            "provider_comparison": {},
            "strategy_analysis": {},
            "recommendations": []
        }
        
        # 提取所有成功的结果
        all_results = []
        for provider, provider_data in comprehensive_results.items():
            if provider_data.get("success", True) and "comparison_results" in provider_data:
                for strategy, strategy_data in provider_data["comparison_results"].items():
                    if strategy_data.get("success", True) and "success_rate" in strategy_data:
                        all_results.append({
                            "provider": provider,
                            "strategy": strategy,
                            **strategy_data
                        })
        
        if all_results:
            # 概览
            success_rates = [r["success_rate"] for r in all_results]
            report["overview"] = {
                "total_experiments": len(all_results),
                "avg_success_rate": sum(success_rates) / len(success_rates),
                "best_combination": max(all_results, key=lambda x: x["success_rate"]),
                "fastest_generation": min(all_results, key=lambda x: x["generation_stats"].get("average_time", float('inf')))
            }
            
            # 提供商比较
            provider_stats = {}
            for provider in set(r["provider"] for r in all_results):
                provider_results = [r for r in all_results if r["provider"] == provider]
                provider_success_rates = [r["success_rate"] for r in provider_results]
                provider_stats[provider] = {
                    "avg_success_rate": sum(provider_success_rates) / len(provider_success_rates),
                    "experiments_count": len(provider_results)
                }
            
            report["provider_comparison"] = provider_stats
            
            # 策略分析
            strategy_stats = {}
            for strategy in set(r["strategy"] for r in all_results):
                strategy_results = [r for r in all_results if r["strategy"] == strategy]
                strategy_success_rates = [r["success_rate"] for r in strategy_results]
                strategy_stats[strategy] = {
                    "avg_success_rate": sum(strategy_success_rates) / len(strategy_success_rates),
                    "experiments_count": len(strategy_results)
                }
            
            report["strategy_analysis"] = strategy_stats
            
            # 生成建议
            best_strategy = max(strategy_stats.items(), key=lambda x: x[1]["avg_success_rate"])
            best_provider = max(provider_stats.items(), key=lambda x: x[1]["avg_success_rate"])
            
            report["recommendations"] = [
                f"推荐使用策略: {best_strategy[0]} (平均成功率: {best_strategy[1]['avg_success_rate']:.2%})",
                f"推荐使用提供商: {best_provider[0]} (平均成功率: {best_provider[1]['avg_success_rate']:.2%})",
                f"最佳组合: {report['overview']['best_combination']['provider']} + {report['overview']['best_combination']['strategy']}"
            ]
        
        return report
    
    def _save_strategy_result(self, result: Dict[str, Any]):
        """保存单一策略结果"""
        filename = f"{result['strategy']}_{result['provider']}_result.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"策略结果已保存: {filepath}")
    
    def _save_comparison_result(self, comparison_results: Dict[str, Any], comparison_report: Dict[str, Any]):
        """保存比较结果"""
        # 保存详细结果
        filepath = self.results_dir / "strategy_comparison.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存比较报告
        report_filepath = self.results_dir / "comparison_report.json"
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"比较结果已保存: {filepath}")
    
    def _save_comprehensive_result(self):
        """保存综合实验结果"""
        filepath = self.results_dir / "comprehensive_experiment.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"综合实验结果已保存: {filepath}")
    
    async def run_quick_test(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        运行快速测试
        
        Args:
            num_samples: 测试样本数量
            
        Returns:
            测试结果
        """
        logger.info(f"开始快速测试，样本数: {num_samples}")
        
        generator = None
        try:
            # 获取测试样本
            samples = self.data_sampler.sample_for_experiment("test", num_samples)
            if not samples:
                raise ValueError("未能获取测试样本")
            
            # 使用基础策略测试
            generator = PersonalizedTitleGenerator()
            
            # 生成一个样本看看效果
            sample = samples[0]
            result = await generator.compare_strategies(
                sample["news_content"],
                sample["user_history"]
            )
            
            test_result = {
                "test_sample": sample,
                "strategy_results": result,
                "total_samples_available": len(samples)
            }
            
            # 保存测试结果
            test_filepath = self.results_dir / "quick_test.json"
            with open(test_filepath, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info("快速测试完成")
            return test_result
            
        except Exception as e:
            logger.error(f"快速测试失败: {e}")
            raise
        finally:
            if generator:
                await generator.close()


async def main():
    """主函数，用于运行实验"""
    # 设置日志
    config_manager.setup_logging()
    
    # 创建实验运行器
    runner = ExperimentRunner("test_run")
    
    try:
        # 运行快速测试
        logger.info("运行快速测试...")
        test_result = await runner.run_quick_test(3)
        
        if test_result:
            logger.info("快速测试成功！")
            
            # 显示测试结果
            for strategy, result in test_result["strategy_results"].items():
                if result.get("success"):
                    logger.info(f"{strategy}: {result['generated_title']}")
                else:
                    logger.error(f"{strategy}: 失败 - {result.get('error')}")
            
            # 如果快速测试成功，可以运行完整实验
            # logger.info("运行完整实验...")
            # await runner.run_comprehensive_experiment(num_samples=20)
        
    except Exception as e:
        logger.error(f"实验运行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())