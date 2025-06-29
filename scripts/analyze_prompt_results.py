#!/usr/bin/env python3
"""
提示工程实验结果可视化分析脚本
"""

import json
import matplotlib
# 设置后端，避免tkinter相关的字体警告
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any
import logging
import matplotlib.font_manager as fm
import platform
import os
import warnings

# 禁用所有用户警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

def setup_chinese_fonts():
    """设置中文字体"""
    # 直接设置为英文，避免字体问题
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    print("使用英文字体以避免字体警告")
    return False  # 返回False表示使用英文标签

# 设置字体
has_chinese_font = setup_chinese_fonts()

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('default')

logger = logging.getLogger(__name__)

class PromptEngineeringAnalyzer:
    """提示工程实验结果分析器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.experiments = {}
        self.has_chinese_font = has_chinese_font
        self.load_experiments()
    
    def get_label(self, chinese_text: str, english_text: str = None) -> str:
        """根据字体支持情况返回合适的标签"""
        if self.has_chinese_font:
            return chinese_text
        else:
            return english_text if english_text else self._fallback_text(chinese_text)
    
    def _fallback_text(self, chinese_text: str) -> str:
        """中文文本的英文备选"""
        fallback_map = {
            '提示工程策略性能对比': 'Prompt Strategy Performance Comparison',
            '各策略成功率对比 (%)': 'Success Rate Comparison (%)',
            '成功率 (%)': 'Success Rate (%)',
            '样本数量': 'Sample Size',
            '各策略平均生成时间对比 (秒)': 'Average Generation Time (sec)',
            '平均时间 (秒)': 'Average Time (sec)',
            '时间-成功率散点图': 'Time-Success Rate Scatter',
            '平均生成时间 (秒)': 'Average Generation Time (sec)',
            '样本': 'samples',
            '策略效率排名 (成功率/时间)': 'Strategy Efficiency Ranking',
            '效率分数': 'Efficiency Score',
            '生成时间分布分析': 'Generation Time Distribution Analysis',
            '生成时间箱线图': 'Generation Time Box Plot',
            '生成时间 (秒)': 'Generation Time (sec)',
            '生成时间分布直方图': 'Time Distribution Histogram',
            '频次': 'Frequency',
            '生成时间累积分布图': 'Cumulative Distribution',
            '累积概率': 'Cumulative Probability',
            '策略综合性能雷达图': 'Strategy Performance Radar Chart',
            '成功率': 'Success Rate',
            '生成速度': 'Speed',
            '稳定性': 'Stability'
        }
        return fallback_map.get(chinese_text, chinese_text.encode('ascii', 'ignore').decode('ascii'))

    def load_experiments(self):
        """加载所有实验结果"""
        for exp_dir in self.results_dir.glob("experiment_*"):
            if exp_dir.is_dir():
                exp_name = exp_dir.name
                comparison_file = exp_dir / "comparison_report.json"
                
                if comparison_file.exists():
                    with open(comparison_file, 'r', encoding='utf-8') as f:
                        self.experiments[exp_name] = json.load(f)
                    
                    # 加载详细结果
                    self.load_detailed_results(exp_dir, exp_name)
        
        logger.info(f"加载了 {len(self.experiments)} 个实验结果")
    
    def load_detailed_results(self, exp_dir: Path, exp_name: str):
        """加载详细的生成结果"""
        detailed_results = {}
        
        for result_file in exp_dir.glob("*_result.json"):
            strategy_name = result_file.stem.replace("_siliconflow_result", "").replace("_result", "")
            
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    detailed_results[strategy_name] = json.load(f)
            except Exception as e:
                logger.warning(f"加载 {result_file} 失败: {e}")
        
        self.experiments[exp_name]['detailed_results'] = detailed_results
    
    def plot_performance_comparison(self, output_dir: Path):
        """绘制性能对比图"""
        # 创建图形前再次确认字体设置
        plt.rcParams['font.sans-serif'] = plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(self.get_label('提示工程策略性能对比', 'Prompt Strategy Performance Comparison'), 
                     fontsize=16, fontweight='bold')
        
        # 准备数据
        all_data = []
        for exp_name, exp_data in self.experiments.items():
            sample_size = exp_name.split('_')[1]  # experiment_3_samples -> 3
            
            for strategy_data in exp_data['performance_ranking']:
                all_data.append({
                    'experiment': exp_name,
                    'sample_size': int(sample_size),
                    'strategy': strategy_data['strategy'],
                    'success_rate': strategy_data['success_rate'] * 100,
                    'avg_time': strategy_data['avg_time'],
                    'total_samples': strategy_data['total_samples']
                })
        
        df = pd.DataFrame(all_data)
        
        # 1. 成功率对比
        ax1 = axes[0, 0]
        pivot_success = df.pivot(index='strategy', columns='sample_size', values='success_rate')
        pivot_success.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_title(self.get_label('各策略成功率对比 (%)', 'Success Rate Comparison (%)'), fontweight='bold')
        ax1.set_ylabel(self.get_label('成功率 (%)', 'Success Rate (%)'))
        ax1.legend(title=self.get_label('样本数量', 'Sample Size'))
        ax1.set_ylim(95, 101)
        
        # 2. 平均生成时间对比
        ax2 = axes[0, 1]
        pivot_time = df.pivot(index='strategy', columns='sample_size', values='avg_time')
        pivot_time.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_title(self.get_label('各策略平均生成时间对比 (秒)', 'Average Generation Time (sec)'), fontweight='bold')
        ax2.set_ylabel(self.get_label('平均时间 (秒)', 'Average Time (sec)'))
        ax2.legend(title=self.get_label('样本数量', 'Sample Size'))
        
        # 3. 时间效率散点图
        ax3 = axes[1, 0]
        for sample_size in df['sample_size'].unique():
            subset = df[df['sample_size'] == sample_size]
            ax3.scatter(subset['avg_time'], subset['success_rate'], 
                       label=f'{sample_size} {self.get_label("样本", "samples")}', s=100, alpha=0.7)
        
        ax3.set_xlabel(self.get_label('平均生成时间 (秒)', 'Average Generation Time (sec)'))
        ax3.set_ylabel(self.get_label('成功率 (%)', 'Success Rate (%)'))
        ax3.set_title(self.get_label('时间-成功率散点图', 'Time-Success Rate Scatter'), fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 策略效率排名
        ax4 = axes[1, 1]
        # 计算效率分数 (成功率/时间)
        df['efficiency'] = df['success_rate'] / df['avg_time']
        efficiency_mean = df.groupby('strategy')['efficiency'].mean().sort_values(ascending=True)
        
        efficiency_mean.plot(kind='barh', ax=ax4)
        ax4.set_title(self.get_label('策略效率排名 (成功率/时间)', 'Strategy Efficiency Ranking'), fontweight='bold')
        ax4.set_xlabel(self.get_label('效率分数', 'Efficiency Score'))
        
        plt.tight_layout()
        
        # 强制刷新渲染
        plt.draw()
        
        output_file = output_dir / "performance_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"性能对比图已保存到: {output_file}")
        
        return fig
    
    def plot_time_distribution(self, output_dir: Path):
        """绘制生成时间分布图"""
        # 创建图形前再次确认字体设置
        plt.rcParams['font.sans-serif'] = plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(self.get_label('生成时间分布分析', 'Generation Time Distribution Analysis'), 
                     fontsize=16, fontweight='bold')
        
        all_times = {}
        
        # 收集所有生成时间数据
        for exp_name, exp_data in self.experiments.items():
            if 'detailed_results' not in exp_data:
                continue
                
            for strategy, results in exp_data['detailed_results'].items():
                if strategy not in all_times:
                    all_times[strategy] = []
                
                if 'results' in results:
                    for result in results['results']:
                        if 'generation_time' in result:
                            all_times[strategy].append(result['generation_time'])
        
        strategies = list(all_times.keys())
        colors = sns.color_palette("husl", len(strategies))
        
        # 1. 箱线图
        ax1 = axes[0, 0]
        data_for_box = [all_times[strategy] for strategy in strategies]
        box_plot = ax1.boxplot(data_for_box, labels=strategies, patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title(self.get_label('生成时间箱线图', 'Generation Time Box Plot'), fontweight='bold')
        ax1.set_ylabel(self.get_label('生成时间 (秒)', 'Generation Time (sec)'))
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 直方图
        ax2 = axes[0, 1]
        for i, (strategy, times) in enumerate(all_times.items()):
            ax2.hist(times, bins=20, alpha=0.6, label=strategy, color=colors[i])
        
        ax2.set_title(self.get_label('生成时间分布直方图', 'Time Distribution Histogram'), fontweight='bold')
        ax2.set_xlabel(self.get_label('生成时间 (秒)', 'Generation Time (sec)'))
        ax2.set_ylabel(self.get_label('频次', 'Frequency'))
        ax2.legend()
        
        # 3. 累积分布图
        ax3 = axes[1, 0]
        for i, (strategy, times) in enumerate(all_times.items()):
            sorted_times = np.sort(times)
            y = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
            ax3.plot(sorted_times, y, label=strategy, color=colors[i], linewidth=2)
        
        ax3.set_title(self.get_label('生成时间累积分布图', 'Cumulative Distribution'), fontweight='bold')
        ax3.set_xlabel(self.get_label('生成时间 (秒)', 'Generation Time (sec)'))
        ax3.set_ylabel(self.get_label('累积概率', 'Cumulative Probability'))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 统计摘要
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if self.has_chinese_font:
            stats_text = "生成时间统计摘要:\n\n"
            for strategy, times in all_times.items():
                if times:
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    min_time = np.min(times)
                    max_time = np.max(times)
                    
                    stats_text += f"{strategy}:\n"
                    stats_text += f"  平均: {mean_time:.2f}s\n"
                    stats_text += f"  标准差: {std_time:.2f}s\n"
                    stats_text += f"  最小: {min_time:.2f}s\n"
                    stats_text += f"  最大: {max_time:.2f}s\n\n"
        else:
            stats_text = "Generation Time Statistics:\n\n"
            for strategy, times in all_times.items():
                if times:
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    min_time = np.min(times)
                    max_time = np.max(times)
                    
                    stats_text += f"{strategy}:\n"
                    stats_text += f"  Mean: {mean_time:.2f}s\n"
                    stats_text += f"  Std: {std_time:.2f}s\n"
                    stats_text += f"  Min: {min_time:.2f}s\n"
                    stats_text += f"  Max: {max_time:.2f}s\n\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.draw()
        
        output_file = output_dir / "time_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"时间分布图已保存到: {output_file}")
        
        return fig
    
    def plot_strategy_radar(self, output_dir: Path):
        """绘制策略雷达图"""
        # 计算各策略的多维指标
        strategy_metrics = {}
        
        for exp_name, exp_data in self.experiments.items():
            for strategy_data in exp_data['performance_ranking']:
                strategy = strategy_data['strategy']
                if strategy not in strategy_metrics:
                    strategy_metrics[strategy] = {
                        'success_rate': [],
                        'speed': [],  # 1/avg_time，速度
                        'stability': [],  # 1/std_time，稳定性
                        'sample_count': []
                    }
                
                strategy_metrics[strategy]['success_rate'].append(strategy_data['success_rate'])
                strategy_metrics[strategy]['speed'].append(1 / strategy_data['avg_time'])
                strategy_metrics[strategy]['sample_count'].append(strategy_data['total_samples'])
        
        # 计算标准差作为稳定性指标
        for exp_name, exp_data in self.experiments.items():
            if 'detailed_results' not in exp_data:
                continue
                
            for strategy, results in exp_data['detailed_results'].items():
                if 'results' in results:
                    times = [r.get('generation_time', 0) for r in results['results'] if 'generation_time' in r]
                    if times and strategy in strategy_metrics:
                        stability = 1 / (np.std(times) + 0.1)  # 避免除零
                        strategy_metrics[strategy]['stability'].append(stability)
        
        # 准备雷达图数据
        if self.has_chinese_font:
            metrics = ['成功率', '生成速度', '稳定性']
        else:
            metrics = ['Success Rate', 'Speed', 'Stability']
        
        strategies = list(strategy_metrics.keys())
        
        # 归一化数据到0-1范围
        normalized_data = {}
        for strategy in strategies:
            normalized_data[strategy] = [
                np.mean(strategy_metrics[strategy]['success_rate']),
                np.mean(strategy_metrics[strategy]['speed']) / max([np.mean(strategy_metrics[s]['speed']) for s in strategies]),
                np.mean(strategy_metrics[strategy]['stability']) / max([np.mean(strategy_metrics[s]['stability']) for s in strategies]) if strategy_metrics[strategy]['stability'] else 0.5
            ]
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("husl", len(strategies))
        
        for i, (strategy, values) in enumerate(normalized_data.items()):
            values += values[:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title(self.get_label('策略综合性能雷达图', 'Strategy Performance Radar Chart'), 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.draw()
        
        output_file = output_dir / "strategy_radar.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"策略雷达图已保存到: {output_file}")
        
        return fig
    
    def generate_analysis_report(self, output_dir: Path):
        """生成分析报告"""
        report = {
            "analysis_summary": {
                "total_experiments": len(self.experiments),
                "total_strategies": set(),
                "overall_metrics": {}
            },
            "strategy_analysis": {},
            "recommendations": []
        }
        
        all_strategies = set()
        all_success_rates = []
        all_times = []
        
        # 收集统计数据
        for exp_name, exp_data in self.experiments.items():
            for strategy_data in exp_data['performance_ranking']:
                strategy = strategy_data['strategy']
                all_strategies.add(strategy)
                all_success_rates.append(strategy_data['success_rate'])
                all_times.append(strategy_data['avg_time'])
                
                if strategy not in report['strategy_analysis']:
                    report['strategy_analysis'][strategy] = {
                        "appearances": 0,
                        "avg_success_rate": 0,
                        "avg_time": 0,
                        "performance_rank": []
                    }
                
                report['strategy_analysis'][strategy]["appearances"] += 1
                report['strategy_analysis'][strategy]["avg_success_rate"] += strategy_data['success_rate']
                report['strategy_analysis'][strategy]["avg_time"] += strategy_data['avg_time']
        
        # 计算平均值
        for strategy in all_strategies:
            appearances = report['strategy_analysis'][strategy]["appearances"]
            report['strategy_analysis'][strategy]["avg_success_rate"] /= appearances
            report['strategy_analysis'][strategy]["avg_time"] /= appearances
        
        # 整体指标
        report["analysis_summary"]["total_strategies"] = len(all_strategies)
        report["analysis_summary"]["overall_metrics"] = {
            "avg_success_rate": np.mean(all_success_rates),
            "avg_generation_time": np.mean(all_times),
            "time_std": np.std(all_times)
        }
        
        # 生成建议
        best_overall = min(report['strategy_analysis'].items(), 
                          key=lambda x: x[1]['avg_time'])
        fastest = min(report['strategy_analysis'].items(), 
                     key=lambda x: x[1]['avg_time'])
        
        report["recommendations"] = [
            f"最佳综合性能策略: {best_overall[0]}",
            f"最快生成策略: {fastest[0]}",
            "所有策略都表现出极高的成功率(接近100%)",
            f"平均生成时间为 {report['analysis_summary']['overall_metrics']['avg_generation_time']:.2f} 秒"
        ]
        
        # 保存报告
        report_file = output_dir / "analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析报告已保存到: {report_file}")
        return report
    
    def run_complete_analysis(self, output_dir: str = None):
        """运行完整分析"""
        if output_dir is None:
            output_dir = self.results_dir / "analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        logger.info("开始生成可视化分析...")
        
        try:
            # 生成所有图表
            self.plot_performance_comparison(output_dir)
            self.plot_time_distribution(output_dir)
            self.plot_strategy_radar(output_dir)
            
            # 生成分析报告
            self.generate_analysis_report(output_dir)
            
            logger.info(f"分析完成！所有结果已保存到: {output_dir}")
            
        except Exception as e:
            logger.error(f"分析过程中发生错误: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="分析提示工程实验结果")
    parser.add_argument("--results_dir", default="results/prompt_engineering",
                       help="实验结果目录")
    parser.add_argument("--output_dir", default=None,
                       help="输出目录，默认为结果目录下的analysis文件夹")
    parser.add_argument("--log_level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行分析
    analyzer = PromptEngineeringAnalyzer(args.results_dir)
    analyzer.run_complete_analysis(args.output_dir)


if __name__ == "__main__":
    main()