#!/usr/bin/env python3
"""
基线模型训练结果可视化分析脚本
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional
import logging
import re
import matplotlib.font_manager as fm

def setup_chinese_fonts():
    """设置中文字体"""
    # 尝试多种中文字体
    chinese_fonts = [
        'SimHei',           # Windows 黑体
        'Microsoft YaHei',  # Windows 微软雅黑
        'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
        'Noto Sans CJK SC',     # Google Noto 中文字体
        'DejaVu Sans',          # DejaVu Sans (fallback)
        'Arial Unicode MS',     # Unicode 字体
        'sans-serif'           # 系统默认
    ]
    
    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        print(f"使用字体: {selected_font}")
    else:
        # 如果没有找到中文字体，使用英文标签
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("警告: 未找到中文字体，将使用英文标签")
        return False
    
    plt.rcParams['axes.unicode_minus'] = False
    return True

# 设置字体
has_chinese_font = setup_chinese_fonts()

# 设置图表样式
sns.set_style("whitegrid")

logger = logging.getLogger(__name__)

class BaselineModelAnalyzer:
    """基线模型训练结果分析器"""
    
    def __init__(self, results_dir: str, logs_dir: str = "logs"):
        self.results_dir = Path(results_dir)
        self.logs_dir = Path(logs_dir)
        self.checkpoints = {}
        self.training_logs = []
        self.has_chinese_font = has_chinese_font
        self.load_data()
    
    def load_data(self):
        """加载训练数据和检查点"""
        # 加载检查点信息
        checkpoint_dir = self.results_dir / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint_file in checkpoint_dir.glob("checkpoint_epoch_*.json"):
                epoch_num = int(re.search(r'epoch_(\d+)', checkpoint_file.name).group(1))
                
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        self.checkpoints[epoch_num] = json.load(f)
                except Exception as e:
                    logger.warning(f"加载检查点 {checkpoint_file} 失败: {e}")
        
        # 加载训练日志
        self.load_training_logs()
        
        logger.info(f"加载了 {len(self.checkpoints)} 个检查点和 {len(self.training_logs)} 条训练记录")
    
    def load_training_logs(self):
        """从日志文件中解析训练信息"""
        log_files = [
            self.logs_dir / "training.log",
            self.logs_dir / "stable_training.log"
        ]
        
        for log_file in log_files:
            if log_file.exists():
                self.parse_log_file(log_file)
    
    def parse_log_file(self, log_file: Path):
        """解析单个日志文件"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # 解析训练损失信息
                    if "Epoch" in line and ("loss" in line.lower() or "Loss" in line):
                        log_entry = self.parse_training_line(line)
                        if log_entry:
                            self.training_logs.append(log_entry)
        except Exception as e:
            logger.warning(f"解析日志文件 {log_file} 失败: {e}")
    
    def parse_training_line(self, line: str) -> Optional[Dict]:
        """解析训练日志行"""
        try:
            # 尝试提取epoch、损失等信息
            epoch_match = re.search(r'Epoch[:\s]+(\d+)', line)
            loss_match = re.search(r'[Ll]oss[:\s]+([\d.]+)', line)
            
            if epoch_match and loss_match:
                return {
                    'epoch': int(epoch_match.group(1)),
                    'loss': float(loss_match.group(1)),
                    'timestamp': line.split()[0] + " " + line.split()[1] if len(line.split()) > 1 else "",
                    'raw_line': line.strip()
                }
        except Exception as e:
            logger.debug(f"解析训练行失败: {e}")
        
        return None
    
    def get_label(self, chinese_text: str, english_text: str = None) -> str:
        """根据字体支持情况返回合适的标签"""
        if self.has_chinese_font:
            return chinese_text
        else:
            return english_text if english_text else chinese_text.encode('ascii', 'ignore').decode('ascii')

    def plot_training_curves(self, output_dir: Path):
        """绘制训练曲线"""
        if not self.training_logs and not self.checkpoints:
            logger.warning("没有找到训练数据，跳过训练曲线绘制")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(self.get_label('基线模型训练分析', 'Baseline Model Training Analysis'), 
                     fontsize=16, fontweight='bold')
        
        # 1. 损失曲线（从日志）
        ax1 = axes[0, 0]
        if self.training_logs:
            df_logs = pd.DataFrame(self.training_logs)
            df_logs = df_logs.groupby('epoch')['loss'].mean().reset_index()
            
            ax1.plot(df_logs['epoch'], df_logs['loss'], 'b-', linewidth=2, marker='o', markersize=4)
            ax1.set_title(self.get_label('训练损失曲线', 'Training Loss Curve'), fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, self.get_label('无训练日志数据', 'No training log data'), 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(self.get_label('训练损失曲线 (无数据)', 'Training Loss Curve (No Data)'), fontweight='bold')
        
        # 2. 检查点性能对比
        ax2 = axes[0, 1]
        if self.checkpoints:
            epochs = sorted(self.checkpoints.keys())
            metrics = {}
            
            for epoch in epochs:
                checkpoint = self.checkpoints[epoch]
                for key, value in checkpoint.items():
                    if isinstance(value, (int, float)) and key not in ['epoch']:
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(value)
            
            # 绘制主要指标
            for metric_name, values in metrics.items():
                if len(values) == len(epochs):
                    ax2.plot(epochs, values, marker='o', label=metric_name, linewidth=2)
            
            ax2.set_title(self.get_label('检查点性能指标', 'Checkpoint Performance Metrics'), fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(self.get_label('指标值', 'Metric Value'))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, self.get_label('无检查点数据', 'No checkpoint data'), 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(self.get_label('检查点性能指标 (无数据)', 'Checkpoint Metrics (No Data)'), fontweight='bold')
        
        # 3. 损失分布统计
        ax3 = axes[1, 0]
        if self.training_logs:
            losses = [log['loss'] for log in self.training_logs]
            ax3.hist(losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(np.mean(losses), color='red', linestyle='--', 
                       label=f'{self.get_label("平均值", "Mean")}: {np.mean(losses):.3f}')
            ax3.axvline(np.median(losses), color='green', linestyle='--', 
                       label=f'{self.get_label("中位数", "Median")}: {np.median(losses):.3f}')
            
            ax3.set_title(self.get_label('损失值分布', 'Loss Distribution'), fontweight='bold')
            ax3.set_xlabel('Loss')
            ax3.set_ylabel(self.get_label('频次', 'Frequency'))
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, self.get_label('无损失数据', 'No loss data'), 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title(self.get_label('损失值分布 (无数据)', 'Loss Distribution (No Data)'), fontweight='bold')
        
        # 4. 训练统计摘要
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if self.has_chinese_font:
            stats_text = "训练统计摘要:\n\n"
            
            if self.training_logs:
                losses = [log['loss'] for log in self.training_logs]
                epochs = [log['epoch'] for log in self.training_logs]
                
                stats_text += f"总训练轮次: {max(epochs) if epochs else 0}\n"
                stats_text += f"记录的损失点: {len(losses)}\n"
                stats_text += f"平均损失: {np.mean(losses):.4f}\n"
                stats_text += f"最低损失: {np.min(losses):.4f}\n"
                stats_text += f"最高损失: {np.max(losses):.4f}\n"
                stats_text += f"损失标准差: {np.std(losses):.4f}\n\n"
            
            if self.checkpoints:
                stats_text += f"保存的检查点: {len(self.checkpoints)}\n"
                stats_text += f"检查点轮次: {sorted(self.checkpoints.keys())}\n"
        else:
            stats_text = "Training Statistics Summary:\n\n"
            
            if self.training_logs:
                losses = [log['loss'] for log in self.training_logs]
                epochs = [log['epoch'] for log in self.training_logs]
                
                stats_text += f"Total Epochs: {max(epochs) if epochs else 0}\n"
                stats_text += f"Loss Records: {len(losses)}\n"
                stats_text += f"Average Loss: {np.mean(losses):.4f}\n"
                stats_text += f"Min Loss: {np.min(losses):.4f}\n"
                stats_text += f"Max Loss: {np.max(losses):.4f}\n"
                stats_text += f"Loss Std: {np.std(losses):.4f}\n\n"
            
            if self.checkpoints:
                stats_text += f"Saved Checkpoints: {len(self.checkpoints)}\n"
                stats_text += f"Checkpoint Epochs: {sorted(self.checkpoints.keys())}\n"
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        output_file = output_dir / "training_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"训练分析图已保存到: {output_file}")
        
        return fig
    
    def plot_model_comparison(self, output_dir: Path):
        """绘制模型对比图"""
        if not self.checkpoints:
            logger.warning("没有检查点数据，跳过模型对比")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(self.get_label('模型检查点对比分析', 'Model Checkpoint Comparison Analysis'), 
                     fontsize=16, fontweight='bold')
        
        epochs = sorted(self.checkpoints.keys())
        
        # 收集所有数值型指标
        all_metrics = {}
        for epoch in epochs:
            checkpoint = self.checkpoints[epoch]
            for key, value in checkpoint.items():
                if isinstance(value, (int, float)) and key != 'epoch':
                    if key not in all_metrics:
                        all_metrics[key] = {}
                    all_metrics[key][epoch] = value
        
        # 1. 指标变化趋势
        ax1 = axes[0, 0]
        for metric_name, metric_data in all_metrics.items():
            if len(metric_data) > 1:  # 只显示有多个数据点的指标
                sorted_epochs = sorted(metric_data.keys())
                values = [metric_data[e] for e in sorted_epochs]
                ax1.plot(sorted_epochs, values, marker='o', label=metric_name, linewidth=2)
        
        ax1.set_title(self.get_label('指标变化趋势', 'Metric Trends'), fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(self.get_label('指标值', 'Metric Value'))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 最佳检查点对比
        ax2 = axes[0, 1]
        if all_metrics:
            # 选择一个主要指标进行排序（假设是loss相关的）
            main_metric = None
            for metric_name in all_metrics.keys():
                if 'loss' in metric_name.lower():
                    main_metric = metric_name
                    break
            
            if not main_metric and all_metrics:
                main_metric = list(all_metrics.keys())[0]
            
            if main_metric:
                sorted_epochs = sorted(all_metrics[main_metric].items(), key=lambda x: x[1])
                top_epochs = [x[0] for x in sorted_epochs[:5]]  # 前5个最佳
                
                metric_names = list(all_metrics.keys())[:5]  # 最多显示5个指标
                
                x = np.arange(len(top_epochs))
                width = 0.15
                
                for i, metric_name in enumerate(metric_names):
                    values = [all_metrics[metric_name].get(epoch, 0) for epoch in top_epochs]
                    ax2.bar(x + i * width, values, width, label=metric_name, alpha=0.8)
                
                ax2.set_title(self.get_label('最佳检查点性能对比', 'Best Checkpoint Performance'), fontweight='bold')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel(self.get_label('指标值', 'Metric Value'))
                ax2.set_xticks(x + width * (len(metric_names) - 1) / 2)
                ax2.set_xticklabels(top_epochs)
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, self.get_label('无指标数据', 'No metric data'), 
                    ha='center', va='center', transform=ax2.transAxes)
        
        # 3. 指标相关性热图
        ax3 = axes[1, 0]
        if len(all_metrics) > 1:
            # 构建相关性矩阵
            correlation_data = {}
            common_epochs = set.intersection(*[set(metric_data.keys()) for metric_data in all_metrics.values()])
            
            if len(common_epochs) > 1:
                for metric_name, metric_data in all_metrics.items():
                    correlation_data[metric_name] = [metric_data[epoch] for epoch in sorted(common_epochs)]
                
                corr_df = pd.DataFrame(correlation_data)
                correlation_matrix = corr_df.corr()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax3, cbar_kws={'label': self.get_label('相关系数', 'Correlation')})
                ax3.set_title(self.get_label('指标相关性热图', 'Metric Correlation Heatmap'), fontweight='bold')
            else:
                ax3.text(0.5, 0.5, self.get_label('数据不足\n无法计算相关性', 'Insufficient data\nfor correlation'), 
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, self.get_label('指标数量不足\n无法计算相关性', 'Not enough metrics\nfor correlation'), 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # 4. 检查点大小分析
        ax4 = axes[1, 1]
        checkpoint_sizes = []
        checkpoint_epochs = []
        
        checkpoint_dir = self.results_dir / "checkpoints"
        if checkpoint_dir.exists():
            for pth_file in checkpoint_dir.glob("checkpoint_epoch_*.pth"):
                epoch_num = int(re.search(r'epoch_(\d+)', pth_file.name).group(1))
                size_mb = pth_file.stat().st_size / (1024 * 1024)
                checkpoint_sizes.append(size_mb)
                checkpoint_epochs.append(epoch_num)
        
        if checkpoint_sizes:
            ax4.bar(checkpoint_epochs, checkpoint_sizes, alpha=0.7, color='lightblue', edgecolor='navy')
            ax4.set_title(self.get_label('检查点文件大小', 'Checkpoint File Size'), fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel(self.get_label('文件大小 (MB)', 'File Size (MB)'))
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, self.get_label('无检查点文件', 'No checkpoint files'), 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        output_file = output_dir / "model_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"模型对比图已保存到: {output_file}")
        
        return fig
    
    def generate_training_report(self, output_dir: Path):
        """生成训练报告"""
        report = {
            "training_summary": {
                "total_epochs": 0,
                "total_checkpoints": len(self.checkpoints),
                "training_records": len(self.training_logs)
            },
            "loss_analysis": {},
            "checkpoint_analysis": {},
            "recommendations": []
        }
        
        # 训练损失分析
        if self.training_logs:
            losses = [log['loss'] for log in self.training_logs]
            epochs = [log['epoch'] for log in self.training_logs]
            
            report["training_summary"]["total_epochs"] = max(epochs) if epochs else 0
            report["loss_analysis"] = {
                "mean_loss": float(np.mean(losses)),
                "min_loss": float(np.min(losses)),
                "max_loss": float(np.max(losses)),
                "std_loss": float(np.std(losses)),
                "final_loss": losses[-1] if losses else None,
                "convergence_trend": "下降" if len(losses) > 1 and losses[-1] < losses[0] else "未收敛"
            }
        
        # 检查点分析
        if self.checkpoints:
            epochs = sorted(self.checkpoints.keys())
            best_epoch = epochs[-1] if epochs else None
            
            report["checkpoint_analysis"] = {
                "available_epochs": epochs,
                "best_checkpoint": best_epoch,
                "checkpoint_metrics": {}
            }
            
            # 提取检查点指标
            if best_epoch is not None:
                report["checkpoint_analysis"]["checkpoint_metrics"] = self.checkpoints[best_epoch]
        
        # 生成建议
        recommendations = []
        
        if self.training_logs:
            losses = [log['loss'] for log in self.training_logs]
            if len(losses) > 1:
                if losses[-1] > losses[0]:
                    recommendations.append("训练损失未收敛，建议检查学习率或模型配置")
                else:
                    recommendations.append("训练损失呈下降趋势，模型正在学习")
        
        if len(self.checkpoints) > 0:
            recommendations.append(f"已保存 {len(self.checkpoints)} 个检查点，可用于模型恢复")
        else:
            recommendations.append("建议保存更多检查点以便监控训练进度")
        
        report["recommendations"] = recommendations
        
        # 保存报告
        report_file = output_dir / "training_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练报告已保存到: {report_file}")
        return report
    
    def run_complete_analysis(self, output_dir: str = None):
        """运行完整分析"""
        if output_dir is None:
            output_dir = self.results_dir / "analysis"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        logger.info("开始生成基线模型训练分析...")
        
        try:
            # 生成训练曲线
            self.plot_training_curves(output_dir)
            
            # 生成模型对比
            self.plot_model_comparison(output_dir)
            
            # 生成训练报告
            self.generate_training_report(output_dir)
            
            logger.info(f"基线模型分析完成！结果已保存到: {output_dir}")
            
        except Exception as e:
            logger.error(f"分析过程中发生错误: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="分析基线模型训练结果")
    parser.add_argument("--results_dir", default="results/baseline",
                       help="结果目录")
    parser.add_argument("--logs_dir", default="logs",
                       help="日志目录")
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
    analyzer = BaselineModelAnalyzer(args.results_dir, args.logs_dir)
    analyzer.run_complete_analysis(args.output_dir)


if __name__ == "__main__":
    main()