# ==============================================================================
# 设置镜像环境变量（必须在所有import之前）
# ==============================================================================
import os

# 设置 Hugging Face Mirror
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ==============================================================================
# 研究二：提取现代文化背景下的家庭关系嵌入（使用 Hugging Face Mirror）
# 
# 功能说明：
# 1. 使用 transformers 从 Hugging Face Mirror 下载 hfl/chinese-roberta-wwm-ext 模型
# 2. 读取LLM生成的现代关系描述（JSON格式）
# 3. 提取所有12层的嵌入向量（embedding layer + 11个encoder层）
# 4. 提取三种token的嵌入：CLS、MASK、关系名
# 5. 保存为CSV格式，与参考文献格式完全一致
# 
# 输入文件：modern_descriptions.json （格式：{"关系名": "描述文本"}）
# 输出文件夹：bert_embedding_data/
#   - CLS_encoder_layer0.csv 到 CLS_encoder_layer11.csv
#   - MASK_embedding_output.csv
#   - MASK_encoder_layer0.csv 到 MASK_encoder_layer11.csv
#   - 关系_embedding_output.csv
#   - 关系_encoder_layer0.csv 到 关系_encoder_layer11.csv
#   - extraction_log.txt （详细日志）
# ==============================================================================

import torch
from transformers import BertTokenizer, BertForMaskedLM
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys
from datetime import datetime

# ==============================================================================
# 初始化与配置
# ==============================================================================

print("=" * 80)
print("研究二脚本启动：提取所有层嵌入向量（CLS + MASK + 关系名）")
print(f"✓ 镜像站点已设置: {os.environ['HF_ENDPOINT']}")
print("=" * 80)

# 创建输出文件夹
output_dir = 'bert_embedding_data'
os.makedirs(output_dir, exist_ok=True)

# 创建日志文件
log_file = os.path.join(output_dir, 'extraction_log.txt')
def log_print(message, also_print=True):
    """同时写入日志文件和控制台的函数"""
    if also_print:
        print(message)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# 记录开始时间
start_time = datetime.now()
log_print(f"\n{'='*80}")
log_print(f"执行时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"使用镜像站点: {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
log_print(f"{'='*80}\n") 

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_print(f"设备配置: {device}")
if torch.cuda.is_available():
    log_print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    log_print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ==============================================================================
# 步骤一：从 Hugging Face Mirror 加载模型
# ==============================================================================

model_id = 'hfl/chinese-roberta-wwm-ext'
log_print(f"\n正在从 Hugging Face Mirror 下载模型: {model_id}")
log_print("提示：首次运行需要下载模型，请耐心等待...")

try:
    # 直接使用 transformers 从 Hugging Face Mirror 下载和加载
    log_print("正在加载分词器...")
    tokenizer = BertTokenizer.from_pretrained(model_id)
    log_print("✓ 分词器加载成功")
    
    log_print("正在加载模型...")
    model = BertForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    log_print("✓ 模型加载成功")
    
    # 动态获取模型层数信息
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    log_print(f"✓ 模型配置: {num_layers}层编码器, 隐藏维度={hidden_size}")
    
except Exception as e:
    log_print(f"✗ 致命错误：模型加载失败")
    log_print(f"错误详情: {e}")
    log_print("建议检查：")
    log_print("1. 网络连接是否正常")
    log_print("2. 是否能访问 hf-mirror.com")
    log_print("3. 磁盘空间是否充足")
    sys.exit(1)

# ==============================================================================
# 步骤二：加载关系描述数据
# ==============================================================================

description_file = r"D:\\HuaweiMoveData\\Users\\黄奕\\Desktop\\Archive of OSF Storage\\my\\plms\\descriptions\\modern_descriptions.json"
log_print(f"\n正在加载关系描述文件: {description_file}")

if not os.path.exists(description_file):
    log_print(f"✗ 致命错误：未找到文件 '{description_file}'")
    log_print(f"请确保文件 '{description_file}' 位于当前目录：")
    log_print(f"当前工作目录: {os.getcwd()}")
    sys.exit(1)

try:
    with open(description_file, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    if not descriptions or not isinstance(descriptions, dict):
        log_print(f"✗ 致命错误：文件格式错误")
        log_print("期望格式: {\"关系名\": \"描述文本\", ...}")
        sys.exit(1)
    
    log_print(f"✓ 成功加载 {len(descriptions)} 条关系描述")
    
    # 显示前3个示例
    log_print("\n数据格式验证（前3个示例）:")
    for i, (rel, desc) in enumerate(list(descriptions.items())[:3]):
        preview = desc[:40] + "..." if len(desc) > 40 else desc
        log_print(f"  {i+1}. {rel}: {preview}")
    
except Exception as e:
    log_print(f"✗ 致命错误：读取文件失败 - {e}")
    sys.exit(1)

# ==============================================================================
# 步骤三：配置提示语模板
# ==============================================================================

prompt_template = "这段{relation}的最显著特征是[MASK]。"
log_print(f"\n提示语模板: \"{prompt_template}\"")

# ==============================================================================
# 步骤四：核心功能 - 提取所有层的嵌入向量
# ==============================================================================

log_print("\n" + "="*80)
log_print(f"开始提取嵌入向量（CLS + MASK + 关系名，共{num_layers}层）...")
log_print("="*80)

# 初始化存储结构
cls_embeddings = {f'layer{i}': [] for i in range(num_layers)}
mask_embeddings = {f'layer{i}': [] for i in range(num_layers)}
mask_embeddings['output'] = []

relation_embeddings = {f'layer{i}': [] for i in range(num_layers)}
relation_embeddings['output'] = []

relation_names = []
skipped_relations = []
error_relations = []

with torch.no_grad():
    for rel, desc in tqdm(descriptions.items(), desc="处理关系", ncols=100):
        try:
            # 构造输入文本
            desc_clean = desc.rstrip('。！？,.!?;；')
            text = f"{desc_clean}。{prompt_template.format(relation=rel)}"
            
            # 分词和编码
            inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding=False)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            
            # 前向传播，获取所有层的隐藏状态
            outputs = model(**inputs, output_hidden_states=True)
            
            # 找到各种token的位置
            input_ids = inputs['input_ids'][0]
            cls_index = 0  # CLS token 总是在位置0
            
            # 找到 MASK token 的位置
            mask_indices = torch.where(input_ids == tokenizer.mask_token_id)[0]
            if len(mask_indices) == 0:
                skipped_relations.append(rel)
                log_print(f"  ⚠ 跳过 '{rel}'：未找到 [MASK] token", also_print=False)
                continue
            mask_index = mask_indices[0].item()
            
            # 找到关系名token的位置
            relation_tokens = tokenizer.encode(rel, add_special_tokens=False)
            relation_index = None
            if len(relation_tokens) > 0:
                relation_token_id = relation_tokens[0]
                relation_token_positions = torch.where(input_ids == relation_token_id)[0]
                if len(relation_token_positions) > 0:
                    relation_index = relation_token_positions[0].item()
            
            # 提取所有层的嵌入向量
            # outputs.hidden_states[0] 是 embedding 层
            # outputs.hidden_states[1] 到 outputs.hidden_states[12] 是 encoder 的12层
            all_hidden_states = outputs.hidden_states

            for layer_idx in range(num_layers):
                # encoder layer 0 对应 hidden_states[1]，以此类推
                hidden_state = all_hidden_states[layer_idx + 1] 
                
                # 提取 CLS token 嵌入
                cls_emb = hidden_state[0, cls_index, :].cpu().numpy()
                cls_embeddings[f'layer{layer_idx}'].append(cls_emb)
                
                # 提取 MASK token 嵌入
                mask_emb = hidden_state[0, mask_index, :].cpu().numpy()
                mask_embeddings[f'layer{layer_idx}'].append(mask_emb)
                
                # 提取关系名token嵌入
                if relation_index is not None:
                    rel_emb = hidden_state[0, relation_index, :].cpu().numpy()
                    relation_embeddings[f'layer{layer_idx}'].append(rel_emb)
                else:
                    # 如果找不到关系名token，用零向量填充
                    relation_embeddings[f'layer{layer_idx}'].append(np.zeros(hidden_size))
            
            # 提取最终输出层的嵌入（用于 embedding_output.csv）
            final_hidden_state = all_hidden_states[-1]  # 最后一层
            
            mask_output = final_hidden_state[0, mask_index, :].cpu().numpy()
            mask_embeddings['output'].append(mask_output)
            
            if relation_index is not None:
                rel_output = final_hidden_state[0, relation_index, :].cpu().numpy()
                relation_embeddings['output'].append(rel_output)
            else:
                relation_embeddings['output'].append(np.zeros(hidden_size))
            
            relation_names.append(rel)
            
        except Exception as e:
            error_relations.append((rel, str(e)))
            log_print(f"  ✗ 错误处理 '{rel}': {e}", also_print=False)
            continue

# ==============================================================================
# 步骤五：保存为CSV文件
# ==============================================================================

log_print("\n" + "="*80)
log_print("提取完成，正在保存CSV文件...")
log_print("="*80)

if not relation_names:
    log_print("✗ 致命错误：未成功提取任何嵌入向量")
    log_print("请检查输入数据和提示语模板是否正确")
    sys.exit(1)

try:
    # 列名使用数字索引（0, 1, 2, ..., hidden_size-1）
    column_names = list(range(hidden_size))
    
    # 保存 CLS token 的嵌入向量
    log_print("\n保存CLS token的嵌入向量...")
    for layer_idx in range(num_layers):
        df = pd.DataFrame(cls_embeddings[f'layer{layer_idx}'], index=relation_names, columns=column_names)
        df.index.name = 'word'
        filename = os.path.join(output_dir, f'CLS_encoder_layer{layer_idx}.csv')
        df.to_csv(filename, encoding='utf-8-sig')
        log_print(f"  ✓ {filename}")
    
    # 保存 MASK token 的嵌入向量
    log_print("\n保存MASK token的嵌入向量...")
    for layer_idx in range(num_layers):
        df = pd.DataFrame(mask_embeddings[f'layer{layer_idx}'], index=relation_names, columns=column_names)
        df.index.name = 'word'
        filename = os.path.join(output_dir, f'MASK_encoder_layer{layer_idx}.csv')
        df.to_csv(filename, encoding='utf-8-sig')
        log_print(f"  ✓ {filename}")
    
    # 保存 MASK token 的最终输出
    df = pd.DataFrame(mask_embeddings['output'], index=relation_names, columns=column_names)
    df.index.name = 'word'
    filename = os.path.join(output_dir, 'MASK_embedding_output.csv')
    df.to_csv(filename, encoding='utf-8-sig')
    log_print(f"  ✓ {filename}")
    
    # 保存关系名token的嵌入向量
    log_print("\n保存关系名token的嵌入向量...")
    for layer_idx in range(num_layers):
        df = pd.DataFrame(relation_embeddings[f'layer{layer_idx}'], index=relation_names, columns=column_names)
        df.index.name = 'word'
        filename = os.path.join(output_dir, f'关系_encoder_layer{layer_idx}.csv')
        df.to_csv(filename, encoding='utf-8-sig')
        log_print(f"  ✓ {filename}")
    
    # 保存关系名token的最终输出
    df = pd.DataFrame(relation_embeddings['output'], index=relation_names, columns=column_names)
    df.index.name = 'word'
    filename = os.path.join(output_dir, '关系_embedding_output.csv')
    df.to_csv(filename, encoding='utf-8-sig')
    log_print(f"  ✓ {filename}")
    
except Exception as e:
    log_print(f"✗ 保存文件时出错: {e}")
    sys.exit(1)

# ==============================================================================
# 步骤六：生成详细报告
# ==============================================================================

log_print("\n" + "="*80)
log_print("执行摘要")
log_print("="*80)
log_print(f"使用模型: {model_id}")
log_print(f"模型配置: {num_layers}层编码器, {hidden_size}维隐藏层")
log_print(f"输入关系总数: {len(descriptions)}")
log_print(f"成功提取数量: {len(relation_names)}")
log_print(f"跳过数量: {len(skipped_relations)} (未找到[MASK])")
log_print(f"错误数量: {len(error_relations)}")

# 计算文件统计
total_files = num_layers * 3 + 2  # CLS: 12文件, MASK: 13文件, 关系: 13文件
log_print(f"\n生成CSV文件总数: {total_files}")
log_print(f"  - CLS: {num_layers}个文件 (layer0-layer{num_layers-1})")
log_print(f"  - MASK: {num_layers + 1}个文件 (layer0-layer{num_layers-1} + embedding_output)")
log_print(f"  - 关系名: {num_layers + 1}个文件 (layer0-layer{num_layers-1} + embedding_output)")

# 显示跳过的关系
if skipped_relations:
    log_print(f"\n⚠ 以下 {len(skipped_relations)} 个关系被跳过（未找到[MASK]）:")
    for i, rel in enumerate(skipped_relations[:10], 1):
        log_print(f"  {i}. {rel}")
    if len(skipped_relations) > 10:
        log_print(f"  ... 还有 {len(skipped_relations)-10} 个")

# 显示错误的关系
if error_relations:
    log_print(f"\n✗ 以下 {len(error_relations)} 个关系发生错误:")
    for i, (rel, err) in enumerate(error_relations[:5], 1):
        log_print(f"  {i}. {rel}: {err}")
    if len(error_relations) > 5:
        log_print(f"  ... 还有 {len(error_relations)-5} 个")

# 计算执行时间
end_time = datetime.now()
duration = end_time - start_time
log_print(f"\n执行结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"总耗时: {duration.total_seconds():.2f} 秒")

# 输出文件位置信息
log_print("\n" + "="*80)
log_print("研究二脚本执行完毕！")
log_print("="*80)
log_print(f"所有文件已保存至文件夹: {os.path.abspath(output_dir)}/")
log_print(f"详细日志: {os.path.abspath(log_file)}")
log_print("="*80)