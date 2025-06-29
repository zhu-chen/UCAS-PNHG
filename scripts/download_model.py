from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import sentencepiece  # 确保能导入

# 下载 Bart 模型和分词器
bart_model_name = "facebook/bart-base"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)

# 保存 Bart 模型和分词器到本地目录
bart_save_dir = "./local_bart_model"
bart_tokenizer.save_pretrained(bart_save_dir)
bart_model.save_pretrained(bart_save_dir)

# 下载 Flan-T5 模型和分词器
flan_t5_model_name = "google/flan-t5-base"
flan_t5_tokenizer = T5Tokenizer.from_pretrained(flan_t5_model_name)
flan_t5_model = T5ForConditionalGeneration.from_pretrained(flan_t5_model_name)

# 保存 Flan-T5 模型和分词器到本地目录
flan_t5_save_dir = "./local_flan_t5_model"
flan_t5_tokenizer.save_pretrained(flan_t5_save_dir)
flan_t5_model.save_pretrained(flan_t5_save_dir)