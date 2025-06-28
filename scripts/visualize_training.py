#!/usr/bin/env python3
"""
训练可视化模块
生成训练过程中loss和metrics随epoch变化的图表
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import torch

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
sns.set_style("whitegrid")
sns.set_palette("husl")

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.baseline.utils.logger import setup_logger

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, output_dir: str = "results/visualizations"):
        """初始化可视化器"""
        self.logger = setup_logger(
            name='training_visualizer',
            level=logging.INFO,
            log_file='logs/training_visualization.log',
            console_output=True
        )
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"训练可视化器初始化完成，输出目录: {self.output_dir}")
    
    def load_training_history_from_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, List]]:
        """从检查点文件中加载训练历史"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'train_history' in checkpoint:
                return checkpoint['train_history']
            else:
                self.logger.warning("检查点中未找到训练历史数据")
                return None
                
        except Exception as e:
            self.logger.error(f"加载检查点失败: {str(e)}")
            return None
    
    def load_training_history_from_json(self, history_path: str) -> Optional[Dict[str, List]]:
        """从JSON文件中加载训练历史"""
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"加载训练历史JSON文件失败: {str(e)}")
            return None
    
    def extract_history_from_logs(self, log_path: str) -> Dict[str, List]:
        """从训练日志中提取训练历史"""
        train_losses = []
        valid_losses = []
        rouge_scores = []
        bleu_scores = []
        epochs = []
        
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_epoch = 0
            for line in lines:
                # 解析epoch信息
                if "Train Loss:" in line and "Valid Loss:" in line:
                    try:
                        # 示例格式: "Epoch 1/3 - Train Loss: 2.4567, Valid Loss: 2.3456, ROUGE: 0.1234, BLEU: 0.0567"
                        parts = line.strip().split(" - ")
                        if len(parts) >= 2:
                            epoch_part = parts[0]
                            metrics_part = parts[1]
                            
                            # 提取epoch
                            if "Epoch" in epoch_part:
                                epoch_str = epoch_part.split("/")[0].split()[-1]
                                current_epoch = int(epoch_str)
                            
                            # 提取指标
                            metrics = {}
                            for metric in metrics_part.split(", "):
                                if ": " in metric:
                                    key, value = metric.split(": ")
                                    key = key.strip()
                                    value = float(value.strip())
                                    metrics[key] = value
                            
                            if "Train Loss" in metrics:
                                epochs.append(current_epoch)
                                train_losses.append(metrics["Train Loss"])
                                valid_losses.append(metrics.get("Valid Loss", 0.0))
                                rouge_scores.append(metrics.get("ROUGE", 0.0))
                                bleu_scores.append(metrics.get("BLEU", 0.0))
                                
                    except Exception as e:
                        self.logger.debug(f"解析日志行失败: {line.strip()}, 错误: {str(e)}")
                        continue
            
            self.logger.info(f"从日志中提取到 {len(train_losses)} 个epoch的训练历史")
            
            return {
                'epochs': epochs,
                'train_loss': train_losses,
                'valid_loss': valid_losses,
                'rouge_scores': rouge_scores,
                'bleu_scores': bleu_scores
            }
            
        except Exception as e:
            self.logger.error(f"从日志提取训练历史失败: {str(e)}")
            return {
                'epochs': [],
                'train_loss': [],
                'valid_loss': [],
                'rouge_scores': [],
                'bleu_scores': []
            }
    
    def create_loss_plot(self, history: Dict[str, List], save_path: str = None) -> str:
        """创建训练和验证损失折线图"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        epochs = history.get('epochs', list(range(1, len(history.get('train_loss', [])) + 1)))
        train_loss = history.get('train_loss', [])
        valid_loss = history.get('valid_loss', [])
        
        if not train_loss:
            self.logger.warning("未找到训练损失数据")
            return None
        
        # 绘制训练损失
        ax.plot(epochs, train_loss, 'b-o', label='训练损失', linewidth=2, markersize=6)
        
        # 绘制验证损失
        if valid_loss:
            ax.plot(epochs, valid_loss, 'r-s', label='验证损失', linewidth=2, markersize=6)
        
        # 设置图表属性
        ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
        ax.set_ylabel('损失值 (Loss)', fontsize=12)
        ax.set_title('PENS模型训练过程 - 损失变化曲线', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标注
        for i, (e, tl) in enumerate(zip(epochs, train_loss)):
            if i % max(1, len(epochs)//5) == 0:  # 每隔几个点标注一次
                ax.annotate(f'{tl:.3f}', (e, tl), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
        
        # 保存图片
        if save_path is None:
            save_path = str(self.output_dir / "training_loss_curve.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"损失曲线图已保存到: {save_path}")
        return save_path
    
    def create_metrics_plot(self, history: Dict[str, List], save_path: str = None) -> str:
        """创建评估指标折线图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        epochs = history.get('epochs', list(range(1, len(history.get('rouge_scores', [])) + 1)))
        rouge_scores = history.get('rouge_scores', [])
        bleu_scores = history.get('bleu_scores', [])
        
        # ROUGE得分图
        if rouge_scores:
            ax1.plot(epochs, rouge_scores, 'g-o', label='ROUGE得分', linewidth=2, markersize=6)
            ax1.set_xlabel('训练轮次 (Epoch)', fontsize=12)
            ax1.set_ylabel('ROUGE得分', fontsize=12)
            ax1.set_title('ROUGE评估指标变化', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # 标注最高点
            if rouge_scores:
                max_rouge = max(rouge_scores)
                max_epoch = epochs[rouge_scores.index(max_rouge)]
                ax1.annotate(f'最高: {max_rouge:.4f}', 
                           xy=(max_epoch, max_rouge), 
                           xytext=(max_epoch, max_rouge + max_rouge*0.1),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10, ha='center')
        
        # BLEU得分图
        if bleu_scores:
            ax2.plot(epochs, bleu_scores, 'm-s', label='BLEU得分', linewidth=2, markersize=6)
            ax2.set_xlabel('训练轮次 (Epoch)', fontsize=12)
            ax2.set_ylabel('BLEU得分', fontsize=12)
            ax2.set_title('BLEU评估指标变化', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # 标注最高点
            if bleu_scores:
                max_bleu = max(bleu_scores)
                max_epoch = epochs[bleu_scores.index(max_bleu)]
                ax2.annotate(f'最高: {max_bleu:.4f}', 
                           xy=(max_epoch, max_bleu), 
                           xytext=(max_epoch, max_bleu + max_bleu*0.1),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10, ha='center')
        
        # 保存图片
        if save_path is None:
            save_path = str(self.output_dir / "training_metrics_curve.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"评估指标曲线图已保存到: {save_path}")
        return save_path
    
    def create_combined_plot(self, history: Dict[str, List], save_path: str = None) -> str:
        """创建综合图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = history.get('epochs', list(range(1, len(history.get('train_loss', [])) + 1)))
        train_loss = history.get('train_loss', [])
        valid_loss = history.get('valid_loss', [])
        rouge_scores = history.get('rouge_scores', [])
        bleu_scores = history.get('bleu_scores', [])
        
        # 1. 训练损失
        if train_loss:
            ax1.plot(epochs, train_loss, 'b-o', linewidth=2)
            ax1.set_title('训练损失', fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
        
        # 2. 验证损失
        if valid_loss:
            ax2.plot(epochs, valid_loss, 'r-s', linewidth=2)
            ax2.set_title('验证损失', fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
        
        # 3. ROUGE得分
        if rouge_scores:
            ax3.plot(epochs, rouge_scores, 'g-o', linewidth=2)
            ax3.set_title('ROUGE得分', fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('ROUGE')
            ax3.grid(True, alpha=0.3)
        
        # 4. BLEU得分
        if bleu_scores:
            ax4.plot(epochs, bleu_scores, 'm-s', linewidth=2)
            ax4.set_title('BLEU得分', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('BLEU')
            ax4.grid(True, alpha=0.3)
        
        # 总标题
        fig.suptitle('PENS模型训练过程综合分析', fontsize=16, fontweight='bold')
        
        # 保存图片
        if save_path is None:
            save_path = str(self.output_dir / "training_combined_analysis.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"综合分析图已保存到: {save_path}")
        return save_path
    
    def create_summary_report(self, history: Dict[str, List], save_path: str = None) -> str:
        """创建训练总结报告"""
        epochs = history.get('epochs', list(range(1, len(history.get('train_loss', [])) + 1)))
        train_loss = history.get('train_loss', [])
        valid_loss = history.get('valid_loss', [])
        rouge_scores = history.get('rouge_scores', [])
        bleu_scores = history.get('bleu_scores', [])
        
        # 计算统计信息
        report_data = {
            "训练总结": {
                "总训练轮次": len(epochs) if epochs else 0,
                "最终训练损失": train_loss[-1] if train_loss else "N/A",
                "最终验证损失": valid_loss[-1] if valid_loss else "N/A",
                "最佳ROUGE得分": max(rouge_scores) if rouge_scores else "N/A",
                "最佳BLEU得分": max(bleu_scores) if bleu_scores else "N/A",
                "最佳ROUGE轮次": epochs[rouge_scores.index(max(rouge_scores))] if rouge_scores else "N/A",
                "最佳BLEU轮次": epochs[bleu_scores.index(max(bleu_scores))] if bleu_scores else "N/A"
            },
            "损失变化": {
                "训练损失下降": f"{train_loss[0] - train_loss[-1]:.4f}" if len(train_loss) > 1 else "N/A",
                "验证损失下降": f"{valid_loss[0] - valid_loss[-1]:.4f}" if len(valid_loss) > 1 else "N/A",
                "是否过拟合": "是" if (len(train_loss) > 1 and len(valid_loss) > 1 and 
                              train_loss[-1] < train_loss[0] and valid_loss[-1] > valid_loss[0]) else "否"
            }
        }
        
        # 保存报告
        if save_path is None:
            save_path = str(self.output_dir / "training_summary.json")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"训练总结报告已保存到: {save_path}")
        
        # 打印摘要
        print("\n" + "="*50)
        print("训练过程分析摘要")
        print("="*50)
        for category, metrics in report_data.items():
            print(f"\n{category}:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        print("="*50)
        
        return save_path
    
    def visualize_training(self, 
                          checkpoint_path: str = None,
                          history_json_path: str = None,
                          log_path: str = None,
                          create_all: bool = True) -> Dict[str, str]:
        """执行完整的训练可视化"""
        self.logger.info("开始训练过程可视化...")
        
        # 加载训练历史
        history = None
        
        if checkpoint_path and Path(checkpoint_path).exists():
            self.logger.info(f"从检查点加载训练历史: {checkpoint_path}")
            history = self.load_training_history_from_checkpoint(checkpoint_path)
        
        if history is None and history_json_path and Path(history_json_path).exists():
            self.logger.info(f"从JSON文件加载训练历史: {history_json_path}")
            history = self.load_training_history_from_json(history_json_path)
        
        if history is None and log_path and Path(log_path).exists():
            self.logger.info(f"从日志文件提取训练历史: {log_path}")
            history = self.extract_history_from_logs(log_path)
        
        if history is None or not history.get('train_loss'):
            self.logger.error("未能加载到有效的训练历史数据")
            return {}
        
        # 生成图表
        generated_files = {}
        
        if create_all:
            # 创建损失曲线图
            loss_plot = self.create_loss_plot(history)
            if loss_plot:
                generated_files['loss_curve'] = loss_plot
            
            # 创建评估指标图
            metrics_plot = self.create_metrics_plot(history)
            if metrics_plot:
                generated_files['metrics_curve'] = metrics_plot
            
            # 创建综合分析图
            combined_plot = self.create_combined_plot(history)
            if combined_plot:
                generated_files['combined_analysis'] = combined_plot
            
            # 创建总结报告
            summary_report = self.create_summary_report(history)
            if summary_report:
                generated_files['summary_report'] = summary_report
        
        self.logger.info(f"可视化完成，生成了 {len(generated_files)} 个文件")
        return generated_files


def main():
    parser = argparse.ArgumentParser(description='训练过程可视化工具')
    parser.add_argument('--checkpoint', type=str, 
                       default='results/baseline/checkpoints/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--history_json', type=str, 
                       help='训练历史JSON文件路径')
    parser.add_argument('--log_file', type=str, 
                       default='logs/training.log',
                       help='训练日志文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='results/visualizations',
                       help='输出目录')
    parser.add_argument('--create_all', action='store_true', default=True,
                       help='创建所有类型的图表')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = TrainingVisualizer(args.output_dir)
    
    # 执行可视化
    generated_files = visualizer.visualize_training(
        checkpoint_path=args.checkpoint,
        history_json_path=args.history_json,
        log_path=args.log_file,
        create_all=args.create_all
    )
    
    print(f"\n可视化完成! 生成的文件:")
    for file_type, file_path in generated_files.items():
        print(f"  {file_type}: {file_path}")


if __name__ == '__main__':
    main()