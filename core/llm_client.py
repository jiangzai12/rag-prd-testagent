import google.generativeai as genai
import streamlit as st
import json
import re
import sys
import os

# 确保能引用到 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 添加上级目录到 sys.path，以便导入 config 模块
from config.prompts import PromptManager

@st.cache_data(ttl=3600) # 缓存模型列表，1小时更新一次
def get_available_models(api_key):
    """动态获取当前Key可用的所有Chat模型"""
    model_list = []
    try:
        genai.configure(api_key=api_key)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if "gemini" in m.name.lower():
                    model_list.append(m.name)
        model_list.sort()
        if not model_list:
            model_list = ["models/gemini-1.5-pro"]
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return ["models/gemini-1.5-pro"]
    return model_list

def extract_json_from_text(text):
    """从 AI 的对话回复中提取 JSON 代码块"""
    parsed, _ = extract_json_with_error(text)
    return parsed

def extract_json_with_error(text):
    """提取 JSON，并返回失败原因以便上层做自动修复。"""
    direct_error = ""
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        direct_error = f"完整响应不是合法 JSON: {str(e)}"

    pattern = r"```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        longest_match = max(matches, key=len)
        try:
            return json.loads(longest_match.strip()), None
        except json.JSONDecodeError as e:
            return None, f"代码块 JSON 解析失败: {str(e)}"
            
    if direct_error:
        return None, direct_error
    return None, "响应中未找到可解析的 JSON 数据"

def repair_test_case_json_once(api_key, model_name, raw_response_text, validation_errors):
    """在结构化校验失败后进行一次修复重试。"""
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(model_name)
        errors_text = "\n".join([f"- {err}" for err in validation_errors[:12]])
        truncated_raw = str(raw_response_text)[:12000]
        repair_prompt = f"""
你是一个严格的 JSON 修复器。请将输入修复为“纯合法 JSON 数组”，且每一条用例都必须包含以下字段：
id, module, precondition, step, expected, priority, design_strategy。

修复要求：
1. 仅输出 JSON 数组，不要输出任何解释、Markdown 或多余文本。
2. 所有字段必须是字符串类型。
3. priority 只能是 P0/P1/P2。
4. id 不可重复。
5. step 和 expected 不能为空字符串。

已知问题：
{errors_text if errors_text else '- 无'}

待修复内容：
{truncated_raw}
"""
        response = model.generate_content(repair_prompt)
        repaired_data, parse_error = extract_json_with_error(response.text)
        return repaired_data, response.text, parse_error
    except Exception as e:
        return None, "", f"自动修复调用失败: {str(e)}"

def get_gemini_chat_response(api_key, model_name, history, user_input, system_instruction=None):
    """支持上下文的对话接口"""
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(
            model_name, 
            system_instruction=system_instruction
        )
        chat = model.start_chat(history=history)
        response = chat.send_message(user_input)
        return response.text, chat.history
    except Exception as e:
        error_msg = f"模型调用出错: {str(e)}"
        print(error_msg)
        return error_msg, history

def generate_summary(api_key, content, model_name="models/gemini-1.5-flash"):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        input_str = str(content)[:8000]
        
        # 使用配置中的 Prompt
        prompt = PromptManager.SUMMARY_PROMPT
        
        response = model.generate_content([prompt, input_str])
        return response.text.strip()
    except Exception as e:
        print(f"摘要生成失败: {e}")
        return "未命名业务文档"