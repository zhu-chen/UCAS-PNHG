#!/usr/bin/env python3
"""
提示工程主运行脚本
提供命令行接口来运行个性化新闻标题生成实验
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompt_engineering import (
    ExperimentRunner,
    PersonalizedTitleGenerator,
    template_manager,
    config_manager
)


async def run_demo():
    """运行演示"""
    print("🚀 开始个性化新闻标题生成演示...")
    
    # 示例数据
    news_content = """
    人工智能技术在医疗诊断领域实现重大突破。最新研究显示，由清华大学和北京协和医院联合开发的AI诊断系统，
    能够在30秒内准确识别肺部CT影像中的早期病变，准确率高达96.5%，超过了经验丰富的放射科医生。
    该系统已在多家三甲医院开始试用，有望大幅提升医疗诊断效率，特别是在医疗资源相对匮乏的地区。
    """
    
    user_history = [
        "AI技术助力医疗行业数字化转型",
        "机器学习在疾病预测中的应用前景",
        "科技创新推动精准医疗发展",
        "人工智能改变传统医疗模式",
        "智能诊断系统提高医疗效率"
    ]
    
    try:
        generator = PersonalizedTitleGenerator()
        
        print("\n📝 正在生成个性化标题...")
        print(f"新闻内容摘要: {news_content[:100]}...")
        print(f"用户历史兴趣: {len(user_history)} 条历史标题")
        
        # 比较不同策略
        results = await generator.compare_strategies(news_content, user_history)
        
        print("\n✨ 不同策略生成结果:")
        print("-" * 60)
        
        for strategy, result in results.items():
            strategy_info = template_manager.get_strategy_info(strategy)
            print(f"\n🎯 {strategy_info.get('name', strategy)}")
            print(f"   描述: {strategy_info.get('description', '无描述')}")
            
            if result.get('success'):
                print(f"   生成标题: {result['generated_title']}")
                print(f"   生成时间: {result['generation_time']:.2f}s")
            else:
                print(f"   ❌ 生成失败: {result.get('error', '未知错误')}")
        
        print("\n" + "=" * 60)
        
        # 显示统计信息
        stats = generator.get_statistics()
        print(f"📊 生成统计:")
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   成功数: {stats['successful_generations']}")
        print(f"   平均耗时: {stats['average_time']:.2f}s")
        
        await generator.close()
        
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        return False
    
    return True


async def run_test(num_samples=5):
    """运行快速测试"""
    print(f"🧪 开始快速测试 (样本数: {num_samples})...")
    
    try:
        runner = ExperimentRunner("quick_test")
        result = await runner.run_quick_test(num_samples)
        
        if result:
            print("\n✅ 测试成功完成!")
            print(f"📂 结果已保存到: {runner.results_dir}")
            
            # 显示测试样本信息
            sample = result["test_sample"]
            print(f"\n📰 测试样本:")
            print(f"   用户ID: {sample['user_id']}")
            print(f"   历史标题数: {len(sample['user_history'])}")
            print(f"   原始标题: {sample['original_title']}")
            
            print(f"\n🎯 各策略生成结果:")
            for strategy, strategy_result in result["strategy_results"].items():
                if strategy_result.get('success'):
                    print(f"   {strategy}: {strategy_result['generated_title']}")
                else:
                    print(f"   {strategy}: ❌ {strategy_result.get('error', '失败')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False


async def run_experiment(strategies=None, num_samples=50, providers=None):
    """运行完整实验"""
    print(f"🔬 开始运行完整实验...")
    print(f"   策略: {strategies or '所有可用策略'}")
    print(f"   样本数: {num_samples}")
    print(f"   提供商: {providers or '默认提供商'}")
    
    try:
        runner = ExperimentRunner(f"experiment_{num_samples}_samples")
        
        if strategies and len(strategies) == 1:
            # 单策略实验
            result = await runner.run_single_strategy_experiment(
                strategies[0], num_samples, providers[0] if providers else None
            )
            print(f"\n✅ 单策略实验完成!")
            print(f"   策略: {result['strategy']}")
            print(f"   成功率: {result['success_rate']:.2%}")
            
        elif len(strategies or []) > 1 or not strategies:
            # 策略比较实验
            result = await runner.run_strategy_comparison_experiment(
                strategies, num_samples, providers[0] if providers else None
            )
            print(f"\n✅ 策略比较实验完成!")
            
            # 显示排名
            if "comparison_report" in result:
                ranking = result["comparison_report"].get("performance_ranking", [])
                if ranking:
                    print(f"\n🏆 策略性能排名:")
                    for i, item in enumerate(ranking[:3], 1):
                        print(f"   {i}. {item['strategy']}: {item['success_rate']:.2%}")
        
        print(f"\n📂 完整结果已保存到: {runner.results_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="个性化新闻标题生成 - 提示工程方案",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_prompt_engineering.py demo              # 运行演示
  python run_prompt_engineering.py test              # 快速测试
  python run_prompt_engineering.py test --samples 10 # 使用10个样本测试
  python run_prompt_engineering.py experiment        # 运行完整实验
  python run_prompt_engineering.py experiment --strategies basic chain_of_thought --samples 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行快速测试')
    test_parser.add_argument('--samples', type=int, default=5, help='测试样本数量')
    
    # 实验命令
    exp_parser = subparsers.add_parser('experiment', help='运行实验')
    exp_parser.add_argument('--strategies', nargs='+', 
                           choices=['basic', 'chain_of_thought', 'role_playing', 'few_shot'],
                           help='使用的策略')
    exp_parser.add_argument('--samples', type=int, default=50, help='实验样本数量')
    exp_parser.add_argument('--providers', nargs='+', 
                           choices=['siliconflow', 'openai', 'zhipu'],
                           help='API提供商')
    
    # 列出可用策略
    list_parser = subparsers.add_parser('list', help='列出可用策略')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 显示配置信息
    print("🔧 配置信息:")
    print(f"   主要提供商: {config_manager.get_config('model.primary_provider')}")
    print(f"   默认模型: {config_manager.get_config('model.default_model')}")
    print(f"   可用策略: {', '.join(template_manager.list_strategies())}")
    print()
    
    if args.command == 'demo':
        success = asyncio.run(run_demo())
    elif args.command == 'test':
        success = asyncio.run(run_test(args.samples))
    elif args.command == 'experiment':
        success = asyncio.run(run_experiment(args.strategies, args.samples, args.providers))
    elif args.command == 'list':
        print("📋 可用的提示策略:")
        for strategy in template_manager.list_strategies():
            info = template_manager.get_strategy_info(strategy)
            print(f"   • {strategy}: {info.get('name', '无名称')}")
            print(f"     {info.get('description', '无描述')}")
        success = True
    else:
        parser.print_help()
        success = False
    
    if success:
        print("\n🎉 操作完成!")
    else:
        print("\n💥 操作失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()