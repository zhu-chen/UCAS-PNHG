from datasets import load_dataset
import os
import pandas as pd
from huggingface_hub import hf_hub_download
import json
import pickle
from collections import defaultdict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_pens_dataset():
    """下载PENS数据集并正确处理新闻内容"""
    print("正在下载PENS数据集...")
    
    # 创建数据目录
    os.makedirs("data/raw", exist_ok=True)
    
    # 检查文件是否已存在
    required_files = ['train.tsv', 'valid.tsv', 'personalized_test.tsv', 'news.tsv']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(f"data/raw/{file}"):
            missing_files.append(file)
    
    if missing_files:
        print(f"需要下载缺失的文件: {missing_files}")
        
        # 下载缺失的文件
        for filename in missing_files:
            try:
                print(f"下载 {filename}...")
                hf_hub_download(
                    repo_id="THEATLAS/PENS",
                    filename=filename,
                    repo_type="dataset",
                    local_dir="data/raw"
                )
                print(f"{filename} 下载成功")
            except Exception as e:
                print(f"下载 {filename} 失败: {e}")
    else:
        print("所有必需文件已存在")
    
    # 加载和处理数据
    try:
        print("加载数据文件...")
        
        # 加载新闻内容（这是关键）
        news_df = load_news_content()
        
        # 加载impression数据
        train_df = pd.read_csv("data/raw/train.tsv", sep='\t')
        val_df = pd.read_csv("data/raw/valid.tsv", sep='\t')
        test_df = pd.read_csv("data/raw/personalized_test.tsv", sep='\t')
        
        print(f"数据加载完成:")
        print(f"  - 新闻库: {len(news_df)} 篇文章")
        print(f"  - 训练集: {len(train_df)} 条记录")
        print(f"  - 验证集: {len(val_df)} 条记录")
        print(f"  - 测试集: {len(test_df)} 条记录")
        
        # 检查数据完整性
        check_data_integrity(news_df, train_df, val_df, test_df)
        
        # 保存处理后的数据
        news_df.to_pickle("data/raw/news_corpus.pkl")
        train_df.to_pickle("data/raw/train.pkl")
        val_df.to_pickle("data/raw/validation.pkl")
        test_df.to_pickle("data/raw/test.pkl")
        
        print("数据处理和保存完成！")
        
    except Exception as e:
        print(f"数据处理失败: {e}")
        raise

def load_news_content():
    """加载新闻内容文件 - 优化内存使用"""
    print("加载新闻内容文件...")
    
    try:
        # 先检查文件大小
        file_size = os.path.getsize("data/raw/news.tsv") / (1024 * 1024 * 1024)  # GB
        print(f"新闻文件大小: {file_size:.2f} GB")
        
        # 分块读取大文件
        chunk_size = 10000  # 每次读取10000行
        chunks = []
        
        print("开始分块读取新闻文件...")
        chunk_count = 0
        
        # 使用chunksize参数分块读取
        for chunk in pd.read_csv(
            "data/raw/news.tsv", 
            sep='\t',
            encoding='utf-8',
            dtype=str,
            na_values=[''],
            keep_default_na=False,
            chunksize=chunk_size
        ):
            chunk_count += 1
            print(f"处理第 {chunk_count} 块，包含 {len(chunk)} 行")
            
            # 重命名列
            column_mapping = {
                'News ID': 'news_id',
                'Category': 'category', 
                'Topic': 'topic',
                'Headline': 'title',
                'News body': 'body',
                'Title entity': 'title_entities',
                'Entity content': 'entity_content'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in chunk.columns:
                    chunk = chunk.rename(columns={old_col: new_col})
            
            # 基本清理
            if 'title' in chunk.columns:
                chunk['title'] = chunk['title'].fillna('')
                # 只保留有标题的行
                chunk = chunk[chunk['title'].str.len() > 0]
            
            if 'body' in chunk.columns:
                chunk['body'] = chunk['body'].fillna('')
            
            if 'category' in chunk.columns:
                chunk['category'] = chunk['category'].fillna('unknown')
            
            # 只保留需要的列，减少内存使用
            required_columns = ['news_id', 'title', 'body', 'category']
            available_columns = [col for col in required_columns if col in chunk.columns]
            chunk = chunk[available_columns]
            
            chunks.append(chunk)
            
            # 定期合并chunks，避免内存积累
            if len(chunks) >= 10:  # 每10个chunk合并一次
                print(f"合并前 {len(chunks)} 个块...")
                combined_chunk = pd.concat(chunks, ignore_index=True)
                chunks = [combined_chunk]
                print(f"合并后内存使用: {combined_chunk.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # 最终合并所有chunks
        print("最终合并所有数据块...")
        news_df = pd.concat(chunks, ignore_index=True)
        
        # 删除重复项
        if 'news_id' in news_df.columns:
            original_len = len(news_df)
            news_df = news_df.drop_duplicates(subset=['news_id'])
            print(f"删除重复新闻: {original_len - len(news_df)} 条")
        
        print(f"新闻内容加载完成，共 {len(news_df)} 篇文章")
        print(f"最终内存使用: {news_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"新闻文件列: {list(news_df.columns)}")
        
        return news_df
        
    except Exception as e:
        print(f"加载新闻内容失败: {e}")
        raise

def clean_news_data(news_df):
    """清理新闻数据 - 简化版本"""
    print("清理新闻数据...")
    
    # 基本清理已在load_news_content中完成
    print(f"数据清理完成，剩余 {len(news_df)} 篇文章")
    return news_df

def check_data_integrity(news_df, train_df, val_df, test_df):
    """检查数据完整性 - 优化内存使用"""
    print("检查数据完整性...")
    
    # 收集所有被引用的新闻ID - 使用集合减少内存
    referenced_news_ids = set()
    
    # 定义一个辅助函数来处理ID字符串
    def extract_ids(id_string, separator=' '):
        if pd.notna(id_string) and str(id_string).strip():
            return str(id_string).split(separator)
        return []
    
    print("从训练集收集新闻ID...")
    for _, row in train_df.iterrows():
        # 点击的新闻
        referenced_news_ids.update(extract_ids(row.get('ClicknewsID')))
        # 正面和负面新闻
        referenced_news_ids.update(extract_ids(row.get('pos')))
        referenced_news_ids.update(extract_ids(row.get('neg')))
    
    print("从验证集收集新闻ID...")
    for _, row in val_df.iterrows():
        referenced_news_ids.update(extract_ids(row.get('ClicknewsID')))
        referenced_news_ids.update(extract_ids(row.get('pos')))
        referenced_news_ids.update(extract_ids(row.get('neg')))
    
    print("从测试集收集新闻ID...")
    for _, row in test_df.iterrows():
        referenced_news_ids.update(extract_ids(row.get('clicknewsID'), ','))
        referenced_news_ids.update(extract_ids(row.get('posnewID'), ','))
    
    # 移除空字符串
    referenced_news_ids.discard('')
    
    # 检查新闻库中的新闻ID
    available_news_ids = set(news_df['news_id'].astype(str))
    
    # 计算覆盖率
    missing_news_ids = referenced_news_ids - available_news_ids
    if len(referenced_news_ids) > 0:
        coverage = (len(referenced_news_ids) - len(missing_news_ids)) / len(referenced_news_ids) * 100
    else:
        coverage = 0
    
    print(f"数据完整性检查结果:")
    print(f"  - 被引用的新闻ID总数: {len(referenced_news_ids)}")
    print(f"  - 新闻库中的新闻ID总数: {len(available_news_ids)}")
    print(f"  - 缺失的新闻ID数量: {len(missing_news_ids)}")
    print(f"  - 覆盖率: {coverage:.2f}%")
    
    if len(missing_news_ids) > 0:
        missing_list = list(missing_news_ids)[:10]
        print(f"  - 前10个缺失的新闻ID: {missing_list}")
    
    return coverage

if __name__ == "__main__":
    download_pens_dataset()
