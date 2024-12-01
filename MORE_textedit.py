import os
import json
import random
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer, util
import torch
import nltk
import pandas as pd
from datasets import Dataset, DatasetDict

nltk.download("averaged_perceptron_tagger")
# 设置代理（如有必要）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

# 加载 RoBERTa 模型用于 MLM
mlm_model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)
mlm_model = AutoModelForMaskedLM.from_pretrained(mlm_model_name)
mlm_pipeline = pipeline("fill-mask", model=mlm_model, tokenizer=tokenizer, device="cuda")

# 加载句子相似度模型
similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 加载 GPT-2 模型用于计算困惑度
gpt2_model_name = "gpt2-large"
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to("cuda")

# 扩展后的属性关键词列表
attribute_keywords = {
    "color": [
        "red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "gray", "brown", 
        "violet", "indigo", "turquoise", "magenta", "maroon", "beige", "navy", "teal", "lime", "gold",
        "silver", "bronze", "peach", "lavender", "coral", "emerald", "jade", "ivory", "mint", "salmon",
        "burgundy", "aqua", "cream", "chocolate", "charcoal", "rose", "ruby", "amber", "sapphire",
        "crimson", "cobalt", "plum", "taupe", "umber", "mustard", "sand", "periwinkle", "moss", "fuchsia",
        "orchid", "cerulean", "pearl", "khaki", "topaz", "opal", "smoky", "sepia", "mauve", 
        "blush", "brick", "hazel", "amethyst"
    ],
    "size": [
        "small", "large", "tiny", "huge", "gigantic", "mini", "massive", "medium", "big", "enormous",
        "petite", "colossal", "grand", "micro", "compact", "mammoth", "slim", "bulky", "oversized", 
        "modest", "ample", "miniature", "towering", "spacious", "diminutive", "wee", "vast", "substantial",
        "portly", "sizable", "thin", "thick", "broad", "narrow", "skinny", "chunky", "lightweight", "heavy",
        "gargantuan", "minuscule", "grande", "lean", "sprawling", "wide", "stout", "dense", "bantam", 
        "sturdy", "monumental", "delicate", "featherweight", "immense", "pocket-sized", "solid", "towering",
        "brawny", "whopping", "average-sized", "grandiose"
    ],
    "shape": [
        "round", "square", "triangular", "oval", "rectangular", "circular", "flat", "curved", "hexagonal", 
        "octagonal", "spherical", "cylindrical", "conical", "pyramidal", "oblong", "diamond", "heart-shaped",
        "star-shaped", "spiral", "wavy", "linear", "jagged", "smooth", "sharp", "pointed", "bulbous", "slender",
        "concave", "convex", "hollow", "solid", "geometric", "angular", "boxy", "zigzag", "orbicular",
        "elliptical", "asymmetrical", "tapered", "gourd-shaped", "fan-shaped", "dome-shaped", "cross-shaped",
        "arc", "globular", "cubical", "trihedral", "lobed", "parabolic", "tubular", "crescent-shaped",
        "scalloped", "triangulate", "arc-shaped", "kite-shaped"
    ]
}


# 日志记录函数
def log_modifications(log_file,img_id, original_caption, modified_caption, attribute, replacement):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"图片ID: {img_id}\n")
        f.write(f"原始描述: {original_caption}\n")
        f.write(f"修改后描述: {modified_caption}\n")
        f.write(f"属性: {attribute} -> 替换为: {replacement}\n")
        f.write("-" * 50 + "\n")
# 筛选出符合属性类型的替换词
def filter_replacements_by_type(predictions, attribute_type):
    # 根据属性类型筛选替换词
    valid_replacements = []
    for pred in predictions:
        if pred['token_str'] in attribute_keywords[attribute_type]:  # 仅保留符合原属性类别的替换词
            valid_replacements.append(pred['token_str'])
    return valid_replacements
# # 用于获取符合属性的替换词
def get_attribute_replacements(text, attribute_type, mask_pipeline, top_k=10):
    replacements = []
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    
    # 确保仅对属性类型中的词性为形容词（JJ）的词进行替换
    for token, pos in pos_tags:
        if token in attribute_keywords[attribute_type] and pos.startswith("JJ"):
            masked_text = text.replace(token, mask_pipeline.tokenizer.mask_token, 1)
            predictions = mask_pipeline(masked_text, top_k=top_k)
            # filtered_replacements = filter_replacements_by_type(predictions, attribute_type)
            replacements.extend([pred['token_str'] for pred in predictions if 'token_str' in pred])
            # replacements.extend(filtered_replacements)
    return replacements
# 用于获取符合属性的替换词
# def get_attribute_replacements(text, attribute_type, mask_pipeline, top_k=10):
#     replacements = []
#     tokens = nltk.word_tokenize(text)
#     pos_tags = nltk.pos_tag(tokens)
    
#     # 确保仅对属性类型中的词性为形容词（JJ）的词进行替换
#     for token, pos in pos_tags:
#         if token in attribute_keywords[attribute_type] and pos.startswith("JJ"):
#             masked_text = text.replace(token, mask_pipeline.tokenizer.mask_token, 1)
#             predictions = mask_pipeline(masked_text, top_k=top_k)
            
#             # 优化：直接在生成阶段过滤替换词
#             filtered_replacements = [
#                 pred['token_str'] for pred in predictions
#                 if pred['token_str'] in attribute_keywords[attribute_type]
#             ]
#             replacements.extend(filtered_replacements)
#     return replacements
# 计算困惑度
def calculate_perplexity(text, gpt2_model, gpt2_tokenizer):
    inputs = gpt2_tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = gpt2_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    return torch.exp(loss).item()

# 选择最佳替换
def select_best_replacement(original_caption, replacements, similarity_model, gpt2_model, gpt2_tokenizer, original_attribute):
    best_replacement = None
    best_perplexity = float('inf')
    
    for replacement in replacements:
        modified_caption = original_caption.replace(original_attribute, replacement, 1)
        similarity_score = util.pytorch_cos_sim(
            similarity_model.encode(original_caption), similarity_model.encode(modified_caption)).item()

        if 0.8 < similarity_score < 0.91:
            perplexity = calculate_perplexity(modified_caption, gpt2_model, gpt2_tokenizer)
            if perplexity < best_perplexity:
                best_perplexity = perplexity
                best_replacement = replacement
    return best_replacement

# 批量处理每个 caption 中的属性描述
def modify_captions_batch(batch):
    modified_captions = []
    for img_id,caption in zip( batch['img_id'],batch['caption']):
        original_caption = caption
        modified_caption = caption
        log_details = None
        
        for attribute_type, attributes in attribute_keywords.items():
            for attribute in attributes:
                if attribute in caption:
                    replacements = get_attribute_replacements(caption, attribute_type, mlm_pipeline)
                    best_replacement = select_best_replacement(caption, replacements, similarity_model, gpt2_model, gpt2_tokenizer, attribute)
                    
                    if best_replacement:
                        modified_caption = modified_caption.replace(attribute, best_replacement, 1)
                        log_details = (img_id,original_caption, modified_caption, attribute, best_replacement)
                        break
        modified_captions.append(modified_caption)
        if log_details:
            log_modifications(batch['log_file'][0], *log_details)
    
    return {'modified_caption': modified_captions}

# 加载并处理 JSON 文件
with open(r'D:\研究生阶段\研0\VSCode_workspace\MORE\data\data\MORE\caption_dict.json', 'r', encoding='utf-8') as f:
    caption_data = json.load(f)

# 将数据转换为 datasets 格式
data_entries = []
for img_id, objects in caption_data.items():
    for obj, caption in objects.items():
        data_entries.append({"img_id": img_id, "obj": obj, "caption": caption, "log_file": "./caption_modification_log.txt"})

dataset = Dataset.from_pandas(pd.DataFrame(data_entries))

# 设置日志文件路径
log_file = './caption_modification_log.txt'
open(log_file, 'w').close()  # 清空日志文件

# 处理数据集并进行批量替换
modified_dataset = dataset.map(modify_captions_batch, batched=True, batch_size=32)

# 将处理后的 caption 保存到新 JSON 文件
modified_caption_dict = {}
for entry in modified_dataset:
    img_id = entry["img_id"]
    obj = entry["obj"]
    modified_caption = entry["modified_caption"]

    if img_id not in modified_caption_dict:
        modified_caption_dict[img_id] = {}
    modified_caption_dict[img_id][obj] = modified_caption

output_file = 'caption_modified.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(modified_caption_dict, f, ensure_ascii=False, indent=4)

print(f"已生成修改后的 caption 文件：{output_file}")