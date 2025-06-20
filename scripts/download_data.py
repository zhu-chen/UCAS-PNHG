from datasets import load_dataset
import os
import pandas as pd
from huggingface_hub import hf_hub_download

def download_pens_dataset():
    """下载PENS数据集并分别处理不同配置"""
    print("正在下载PENS数据集...")
    
    # 创建数据目录
    os.makedirs("data/raw", exist_ok=True)
    
    try:
        # 方法1：尝试分别下载不同的配置
        print("尝试下载训练和验证集...")
        
        # 下载原始文件
        train_file = hf_hub_download(
            repo_id="THEATLAS/PENS",
            filename="train.tsv",
            repo_type="dataset",
            local_dir="data/raw"
        )
        
        val_file = hf_hub_download(
            repo_id="THEATLAS/PENS", 
            filename="valid.tsv",
            repo_type="dataset",
            local_dir="data/raw"
        )
        
        test_file = hf_hub_download(
            repo_id="THEATLAS/PENS",
            filename="personalized_test.tsv", 
            repo_type="dataset",
            local_dir="data/raw"
        )
        
        # 读取并检查数据结构
        print("检查数据文件结构...")
        
        train_df = pd.read_csv(train_file, sep='\t', nrows=5)
        val_df = pd.read_csv(val_file, sep='\t', nrows=5) 
        test_df = pd.read_csv(test_file, sep='\t', nrows=5)
        
        print(f"训练集列: {list(train_df.columns)}")
        print(f"验证集列: {list(val_df.columns)}")
        print(f"测试集列: {list(test_df.columns)}")
        
        # 加载完整数据
        train_full = pd.read_csv(train_file, sep='\t')
        val_full = pd.read_csv(val_file, sep='\t')
        test_full = pd.read_csv(test_file, sep='\t')
        
        print(f"训练集大小: {len(train_full)}")
        print(f"验证集大小: {len(val_full)}")
        print(f"测试集大小: {len(test_full)}")
        
        # 保存为pickle格式以便后续处理
        train_full.to_pickle("data/raw/train.pkl")
        val_full.to_pickle("data/raw/validation.pkl") 
        test_full.to_pickle("data/raw/test.pkl")
        
        print("数据集下载并保存完成！")
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("尝试使用替代方法...")
        
        # 方法2：手动下载文件
        try:
            files_to_download = [
                "train.tsv",
                "valid.tsv", 
                "personalized_test.tsv"
            ]
            
            for filename in files_to_download:
                print(f"下载 {filename}...")
                hf_hub_download(
                    repo_id="THEATLAS/PENS",
                    filename=filename,
                    repo_type="dataset",
                    local_dir="data/raw",
                    local_dir_use_symlinks=False
                )
            
            print("所有文件下载完成！")
            
        except Exception as e2:
            print(f"替代方法也失败: {e2}")
            print("请手动从 https://huggingface.co/datasets/THEATLAS/PENS 下载数据集")

if __name__ == "__main__":
    download_pens_dataset()
