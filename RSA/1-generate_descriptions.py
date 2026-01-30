# ==============================================================================
# 使用 DeepSeek API 生成关系描述文本 (v2.2.2 - 最终修正版)
# 作者：[您的名字]
# 日期：[当前日期]
# 描述：根据关系词在Excel中的行位置，自动选择生成现代或古代描述
#       第2-145行(144条数据)生成现代描述，第146-163行(18条数据)生成古代描述
# ==============================================================================

import os
import json
import time
import pandas as pd
from openai import OpenAI

print("脚本启动：准备使用 DeepSeek API 生成关系描述 (v2.2.2)。")

# --- 步骤 1: 设置 API 客户端 ---
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    print("❌ 错误：未找到环境变量 'DEEPSEEK_API_KEY'。")
    print("请按照教程设置环境变量，并重启您的编辑器或终端。")
    exit()

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
print("✅ API 客户端初始化成功。")

# --- 步骤 2: 从Excel文件读取关系词列表 ---
try:
    filename = "D:\\HuaweiMoveData\\Users\\黄奕\\Desktop\\Archive of OSF Storage\\my\\plms\\relationship.xlsx"
    df = pd.read_excel(filename)
    relationships = df['relations'].tolist()
    print(f"✅ 成功从 '{filename}' 文件中加载了 {len(relationships)} 个关系词。")
    print(f"  - 第2-145行(前144条数据)将生成现代描述")
    print(f"  - 第146-163行(后18条数据)将生成古代描述")
except FileNotFoundError:
    print(f"❌ 错误：未找到文件 '{filename}'。请确保该文件与您的Python脚本在同一个文件夹下。")
    exit()
except KeyError:
    print(f"❌ 错误：在 '{filename}' 文件中未找到名为 'relations' 的列。请检查Excel文件列名。")
    exit()
except Exception as e:
    print(f"❌ 读取Excel文件时发生未知错误: {e}")
    exit()

# --- 步骤 3: 定义生成单个描述的核心函数 ---
def get_single_description(relationship, mode):
    """
    为单个关系调用 DeepSeek API 生成描述。
    :param relationship: 单个关系词字符串，例如 "父-子"。
    :param mode: 'modern' 或 'ancient'。
    :return: 描述字符串，如果失败则返回 None。
    """
    if mode == 'modern':
        prompt = f"""
        请你在当代中国的社会文化背景下，为"{relationship}"这个家庭关系撰写一段描述，呈现它的内涵和特点。

        核心要求如下：
        1.  描述需精准、简洁易懂。
        2.  字数在90——110字之间。若最佳描述字数少于或高于此范围，可适当超出。
        3.  请只返回描述文本本身，不要添加任何额外的解释性文字（例如不要说"好的，这是描述："）。
        """
    elif mode == 'ancient':
        prompt = f"""
        请你扮演一个中国文化史专家，对中国古代的家庭结构关系属性进行描述。请务必在中国古代的语境下，为"{relationship}"这个家庭关系撰写一段描述，呈现它的内涵和特点。

        核心要求如下：
        1.  描述需使用精准、严谨的语言。
        2.  请只返回描述文本本身，不要添加任何额外的解释性文字。
        """
    else:
        return None

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位中国历史文化史的专家，你的任务是根据用户的要求，生成精确、客观的关系描述。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,  # 设置为0，确保结果完全确定性
            max_tokens=500
        )
        description = response.choices[0].message.content.strip()
        return description
    except Exception as e:
        print(f"    ❌ API调用失败 for '{relationship}' ({mode}): {e}")
        return None

# --- 步骤 4: 统一处理所有关系，根据位置自动选择模式 ---
def process_all_relationships_unified(relationships_list):
    """
    统一处理所有关系，根据在Excel中的实际行位置自动选择现代或古代模式
    """
    # 初始化两个输出文件
    modern_filename = "modern_descriptions.json"
    ancient_filename = "ancient_descriptions.json"
    
    # 加载已有的描述（断点续传）
    if os.path.exists(modern_filename):
        with open(modern_filename, 'r', encoding='utf-8') as f:
            modern_descriptions = json.load(f)
        print(f"  检测到现代描述文件，已加载 {len(modern_descriptions)} 条。")
    else:
        modern_descriptions = {}
        
    if os.path.exists(ancient_filename):
        with open(ancient_filename, 'r', encoding='utf-8') as f:
            ancient_descriptions = json.load(f)
        print(f"  检测到古代描述文件，已加载 {len(ancient_descriptions)} 条。")
    else:
        ancient_descriptions = {}

    total = len(relationships_list)
    modern_completed = 0
    ancient_completed = 0

    print(f"\n▶️ 开始处理全部 {total} 个关系...")

    for i, relationship in enumerate(relationships_list):
        # 根据索引位置决定模式
        # pandas索引从0开始，所以：
        # 索引0-143 对应Excel第2-145行（现代关系，144条）
        # 索引144-161 对应Excel第146-163行（古代关系，18条）
        if i < 144:  # 前144条：现代关系（索引0-143）
            mode = 'modern'
            current_dict = modern_descriptions
            output_file = modern_filename
            excel_row = i + 2  # 转换为Excel行号
        else:  # 后18条：古代关系（索引144-161）
            mode = 'ancient'
            current_dict = ancient_descriptions
            output_file = ancient_filename
            excel_row = i + 2

        # 检查是否已存在
        if relationship in current_dict:
            print(f"  ({i+1}/{total}) [Excel第{excel_row}行] 跳过: '{relationship}' (已存在-{mode})")
            continue
            
        print(f"  ({i+1}/{total}) [Excel第{excel_row}行] 正在处理: '{relationship}' ({mode}) ...")
        description = get_single_description(relationship, mode)

        if description:
            current_dict[relationship] = description
            
            # 实时保存到对应文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(current_dict, f, ensure_ascii=False, indent=4)
            
            print(f"    ✅ 成功获取并保存: '{relationship}' ({mode})")
            
            if mode == 'modern':
                modern_completed += 1
            else:
                ancient_completed += 1
                
            time.sleep(1)
        else:
            print(f"    ⚠️ 跳过: '{relationship}' (API获取失败，下次运行时将重试)")

    # 最终统计
    print(f"\n🎉 全部任务已完成！")
    print(f"  📊 现代描述 (Excel第2-145行，共144条):")
    print(f"    - 本次新生成: {modern_completed} 条")
    print(f"    - 文件中总计: {len(modern_descriptions)} 条")
    print(f"    - 保存文件: {modern_filename}")
    print(f"  📊 古代描述 (Excel第146-163行，共18条):")
    print(f"    - 本次新生成: {ancient_completed} 条") 
    print(f"    - 文件中总计: {len(ancient_descriptions)} 条")
    print(f"    - 保存文件: {ancient_filename}")

# --- 步骤 5: 执行生成任务 ---
if __name__ == "__main__":
    process_all_relationships_unified(relationships)
    print("\n✅ 研究二数据生成任务全部完成！")