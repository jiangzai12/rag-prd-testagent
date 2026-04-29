import streamlit as st
import pandas as pd
import json
import sys
import os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import setup_proxy
from config.prompts import PromptManager
from ui.sidebar import render_sidebar
from ui.components import display_results
from core.llm_client import (
    get_gemini_chat_response,
    generate_summary,
    extract_json_from_text,
    extract_json_with_error,
    repair_test_case_json_once
)
from core.case_validator import validate_test_cases
from core.dedup_engine import apply_dedup_strategy
from core.rag_engine import RAGEngine
from core.evaluator import Evaluator # 新增引用
from core.regression_runner import run_regression_suite, load_latest_regression_report
from core.rag_offline_eval import (
    run_rag_offline_evaluation,
    load_latest_rag_eval_report,
    list_rag_eval_reports,
    load_rag_eval_report
)

def split_text_and_json(text):
    """分离 AI 回复中的【分析说明】和【JSON数据】"""
    json_data = extract_json_from_text(text)
    if not json_data:
        return text, None
    
    text_stripped = text.strip()
    split_idx_list = text_stripped.find('[')
    split_idx_dict = text_stripped.find('{')
    split_idx = -1
    if split_idx_list != -1 and split_idx_dict != -1:
        split_idx = min(split_idx_list, split_idx_dict)
    elif split_idx_list != -1:
        split_idx = split_idx_list
    elif split_idx_dict != -1:
        split_idx = split_idx_dict
    
    if split_idx > 0:
        explanation = text_stripped[:split_idx].strip()
        explanation = explanation.replace("```json", "").replace("```", "").strip()
        if len(explanation) < 2: 
            explanation = "✅ 已根据指令生成最新测试用例数据（详情请见右侧预览）"
        return explanation, json_data
    
    if text_stripped.startswith("[") or text_stripped.startswith("{") or text_stripped.startswith("```"):
        return "✅ 已根据指令生成最新测试用例数据（详情请见右侧预览）", json_data
        
    return text, json_data

def validate_or_repair_cases(api_key, model_name, response_text):
    """解析、校验并在失败时进行一次自动修复。"""
    parsed_json, parse_error = extract_json_with_error(response_text)
    first_round_errors = []

    if parsed_json is not None:
        is_valid, errors, normalized = validate_test_cases(parsed_json)
        if is_valid:
            dedup_result = apply_dedup_strategy(normalized)
            return dedup_result["cases"], {
                "status": "passed",
                "attempts": 1,
                "errors": [],
                "hard_removed": dedup_result.get("hard_removed", []),
                "semantic_warnings": dedup_result.get("semantic_warnings", [])
            }
        first_round_errors.extend(errors)
    else:
        first_round_errors.append(parse_error or "未能解析出 JSON 数据")

    repaired_json, _, repair_parse_error = repair_test_case_json_once(
        api_key,
        model_name,
        response_text,
        first_round_errors
    )

    if repaired_json is None:
        all_errors = first_round_errors + [repair_parse_error or "自动修复后仍无法解析 JSON"]
        return None, {"status": "failed", "attempts": 2, "errors": all_errors}

    repaired_valid, repaired_errors, repaired_normalized = validate_test_cases(repaired_json)
    if repaired_valid:
        dedup_result = apply_dedup_strategy(repaired_normalized)
        return dedup_result["cases"], {
            "status": "repaired",
            "attempts": 2,
            "errors": first_round_errors,
            "hard_removed": dedup_result.get("hard_removed", []),
            "semantic_warnings": dedup_result.get("semantic_warnings", [])
        }

    all_errors = first_round_errors + ["自动修复后仍未通过结构化校验"] + repaired_errors
    return None, {"status": "failed", "attempts": 2, "errors": all_errors}

def main():
    setup_proxy()
    st.set_page_config(page_title="Auto_prd_test_expert", layout="wide")
    
    # Session 初始化
    if 'messages' not in st.session_state: st.session_state['messages'] = [] 
    if 'gemini_history' not in st.session_state: st.session_state['gemini_history'] = [] 
    if 'res_data' not in st.session_state: st.session_state['res_data'] = None 
    if 'prd_context' not in st.session_state: st.session_state['prd_context'] = "" 
    if 'rag_context' not in st.session_state: st.session_state['rag_context'] = "" 
    if 'rag_sources_display' not in st.session_state: st.session_state['rag_sources_display'] = None
    if 'processed_files' not in st.session_state: st.session_state['processed_files'] = []
    # 新增：评估报告状态
    if 'eval_report' not in st.session_state: st.session_state['eval_report'] = None
    if 'validation_status' not in st.session_state: st.session_state['validation_status'] = None
    if 'dedup_report' not in st.session_state: st.session_state['dedup_report'] = None
    if 'regression_report' not in st.session_state: st.session_state['regression_report'] = None
    if 'rag_eval_report' not in st.session_state: st.session_state['rag_eval_report'] = None
    if 'rag_eval_history' not in st.session_state: st.session_state['rag_eval_history'] = []

    api_key, selected_model = render_sidebar()
    rag_engine = None
    evaluator = None # 初始化评估器

    if api_key:
        try:
            rag_engine = RAGEngine(api_key)
            evaluator = Evaluator(api_key) # 实例化评估器
        except Exception as e:
            st.sidebar.error(f"引擎初始化失败: {e}")

    st.title("🤖 Auto_prd_test_expert")

    tab_work, tab_manage, tab_regression, tab_rag_eval = st.tabs([
        "💬 智能共创工作台",
        "📚 知识库管理",
        "📈 回归看板",
        "🧪 评测看板"
    ])

    # ==================== Tab 1: 共创工作台 ====================
    with tab_work:
        col_chat, col_preview = st.columns([0.4, 0.6], gap="medium")

        # --- 左侧：对话交互区 ---
        with col_chat:
            st.subheader("需求对话")
            
            # 1. 常驻文件上传区
            with st.expander("📂 上传/补充需求文档", expanded=not st.session_state['messages']):
                uploaded_files = st.file_uploader("拖拽文件至此", accept_multiple_files=True, key="chat_uploader")
                
                c1, c2 = st.columns(2)
                use_kb = c1.checkbox("📚 参考技术规范", value=True)
                use_hist = c2.checkbox("🕰️ 参考历史案例", value=True)

                if uploaded_files and rag_engine and api_key:
                    current_file_names = [f.name for f in uploaded_files]
                    if current_file_names != st.session_state['processed_files']:
                        with st.spinner("正在预处理文档并检索知识库..."):
                            preview_txt = ""
                            prompt_content = []
                            for file in uploaded_files:
                                file.seek(0)
                                if "image" in file.type:
                                    img = Image.open(file)
                                    prompt_content.extend([f"图片 {file.name}:", img])
                                    preview_txt += f"[图片 {file.name}] "
                                elif "pdf" in file.type:
                                    prompt_content.extend([f"文档 {file.name}:", {"mime_type": "application/pdf", "data": file.read()}])
                                    preview_txt += f"[PDF {file.name}] "
                                else:
                                    txt = file.read().decode("utf-8")
                                    prompt_content.append(f"文档 {file.name}:\n{txt}")
                                    preview_txt += txt[:500]
                            # 1. 粗筛：检索 RAG (这里稍微放大一点召回范围，比如 n_results=3 或 5，rag_engine里改不改都行，暂保持3)
                            raw_rag_info, sources = rag_engine.search_context(preview_txt, use_history=use_hist, use_knowledge=use_kb)

                            # --- 新增：LLM 细筛 (Filtering) ---
                            final_rag_context = ""
                            if raw_rag_info:
                                with st.spinner("正在进行知识提纯 (去除无关噪音)..."):
                                    # 构造过滤指令
                                    filter_prompt = PromptManager.get_rag_filter_prompt(preview_txt, raw_rag_info)
                                    
                                    # 调用 LLM (建议用 Flash 模型，速度快)
                                    # 注意：这里直接复用 get_gemini_chat_response 或者直接调用 SDK 均可
                                    # 为了方便，假设我们复用现有的 chat 接口，但不带历史记录
                                    filtered_text, _ = get_gemini_chat_response(
                                        api_key, 
                                        selected_model, # 或者强制指定 "models/gemini-1.5-flash" 以提速
                                        [], # 无历史
                                        filter_prompt
                                    )
                                    
                                    # 判断清洗结果
                                    if "无相关参考资料" in filtered_text:
                                        final_rag_context = ""
                                    else:
                                        final_rag_context = filtered_text

                            # 2. 更新 Session
                            st.session_state['rag_context'] = final_rag_context
                            st.session_state['prd_context'] = preview_txt 
                            st.session_state['current_prompt_content'] = prompt_content

                            if sources and final_rag_context:
                                source_list = "\n".join(sources)
                                # 展示时，展示清洗后的纯净内容
                                st.session_state['rag_sources_list'] = f"\n{source_list}\n"
                            else:
                                st.session_state['rag_sources_list'] = "经 AI 分析，知识库中暂无与当前 PRD 强相关的技术规范。"
                                
                            st.session_state['processed_files'] = current_file_names
                            st.toast("✅ 知识库检索完成！")

            # 2. 按钮逻辑
            btn_label = "🚀 开始生成" if not st.session_state['messages'] else "📤 发送补充文件并分析"
            if st.button(btn_label, type="primary", use_container_width=True):
                if not api_key: st.error("请配置 API Key"); st.stop()
                
                if 'current_prompt_content' in st.session_state:
                    with st.spinner(f"正在使用 {selected_model} 分析..."):
                        initial_prompt = PromptManager.get_initial_prompt(
                            st.session_state['prd_context'], 
                            st.session_state['rag_context']
                        )
                        full_payload = initial_prompt + st.session_state.get('current_prompt_content', [])
                        
                        resp_text, updated_history = get_gemini_chat_response(
                            api_key, selected_model, 
                            st.session_state['gemini_history'], 
                            full_payload, 
                            system_instruction=PromptManager.CORE_SYSTEM_PROMPT
                        )
                        
                        st.session_state['gemini_history'] = updated_history
                        st.session_state['messages'].append({"role": "assistant", "content": resp_text})

                        validated_cases, validation_status = validate_or_repair_cases(
                            api_key,
                            selected_model,
                            resp_text
                        )
                        st.session_state['validation_status'] = validation_status
                        if validated_cases is not None:
                            st.session_state['res_data'] = validated_cases
                            st.session_state['dedup_report'] = {
                                "hard_removed": validation_status.get('hard_removed', []),
                                "semantic_warnings": validation_status.get('semantic_warnings', [])
                            }
                            st.session_state['eval_report'] = None # 重新生成后清空旧的评估报告
                            if validation_status['status'] == 'repaired':
                                st.toast("⚠️ 原始输出未通过校验，系统已自动修复并通过")
                            hard_removed = validation_status.get('hard_removed', [])
                            if hard_removed:
                                st.toast(f"🧹 已自动移除 {len(hard_removed)} 条确定性重复用例")
                            semantic_warnings = validation_status.get('semantic_warnings', [])
                            if semantic_warnings:
                                st.info(f"检测到 {len(semantic_warnings)} 条语义近似告警（未自动删除）")
                        else:
                            st.warning("本次输出未通过结构化校验，已保留上一版有效结果。")
                        
                        del st.session_state['current_prompt_content'] 
                        st.rerun()
                else:
                    if not st.session_state['messages']: st.warning("请先上传文件")
                    else: st.info("请在下方输入框继续对话")

            # 3. 聊天流渲染
            chat_container = st.container(height=500)
            with chat_container:
                if st.session_state.get('rag_context'):
                    with st.expander("📚 本次对话参考的知识库片段 (RAG Context)", expanded=False):
                        
                        # 1. 顶部统一显示所有来源 (Header)
                        raw_sources = st.session_state.get('rag_sources_list', '未知来源')
                        header_title = raw_sources.replace("\n- ", "  &  ").replace("- ", "").strip()
                        
                        # 用 caption 或者 markdown 显示在最上方，不占用蓝色框的位置
                        st.markdown(f"**📄 引用来源:** *{header_title}*")
                        
                        # 2. 内容分段渲染 (Body)
                        # 使用双换行符 \n\n 进行切分，通常 LLM 会用空行来区分不同的逻辑段落
                        # 这样可以将一大段文本拆解成几个独立的蓝色卡片，视觉上更轻松
                        fragments = st.session_state['rag_context'].split('<<<RAG_SEP>>>')
                        
                        for frag in fragments:
                            if frag.strip():
                                # 为每一段核心内容生成一个独立的蓝色方块
                                st.info(frag.strip())

                for msg in st.session_state['messages']:
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "user":
                            st.markdown(msg["content"])
                        else:
                            explanation, _ = split_text_and_json(msg["content"])
                            st.markdown(explanation)
                            if "```json" in msg["content"] or "[" in msg["content"]:
                                with st.expander("🔍 查看 JSON 数据", expanded=False):
                                    st.code(msg["content"][-1000:] if len(msg["content"]) > 1000 else msg["content"], language="json")

            # 4. 底部对话输入
            if prompt := st.chat_input("输入指令 (如: '增加几个异常场景')"):
                if not api_key: st.stop()
                st.session_state['messages'].append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("AI 正在思考..."):
                            refine_prompt_str = PromptManager.get_refinement_prompt(
                                prompt, st.session_state['rag_context']
                            )
                            
                            resp_text, updated_history = get_gemini_chat_response(
                                api_key, selected_model, 
                                st.session_state['gemini_history'], 
                                refine_prompt_str,
                                system_instruction=PromptManager.CORE_SYSTEM_PROMPT
                            )
                            
                            explanation, _ = split_text_and_json(resp_text)
                            st.markdown(explanation)

                            validated_cases, validation_status = validate_or_repair_cases(
                                api_key,
                                selected_model,
                                resp_text
                            )
                            st.session_state['validation_status'] = validation_status
                            if validated_cases is not None:
                                st.session_state['dedup_report'] = {
                                    "hard_removed": validation_status.get('hard_removed', []),
                                    "semantic_warnings": validation_status.get('semantic_warnings', [])
                                }
                                with st.expander("查看数据详情"):
                                    if validation_status['status'] == 'repaired':
                                        st.caption("原始输出未通过校验，已自动修复并更新数据")
                                    else:
                                        st.caption("数据已更新并通过结构化校验")
                                    hard_removed = validation_status.get('hard_removed', [])
                                    if hard_removed:
                                        st.caption(f"已自动硬去重 {len(hard_removed)} 条")
                                    semantic_warnings = validation_status.get('semantic_warnings', [])
                                    if semantic_warnings:
                                        st.caption(f"检测到 {len(semantic_warnings)} 条语义近似告警（仅提示）")
                            
                            st.session_state['gemini_history'] = updated_history
                            st.session_state['messages'].append({"role": "assistant", "content": resp_text})
                            
                            if validated_cases is not None:
                                st.session_state['res_data'] = validated_cases
                                st.session_state['eval_report'] = None # 数据更新后清空评估
                                st.rerun()
                            else:
                                st.error("结构化校验失败，未更新右侧结果。")
                                with st.expander("查看校验失败详情", expanded=False):
                                    for err in validation_status.get('errors', []):
                                        st.markdown(f"- {err}")

        # --- 右侧：预览、归档与评估 ---
        with col_preview:
            st.subheader("📄 实时结果预览")

            validation_status = st.session_state.get('validation_status')
            if validation_status:
                status = validation_status.get('status')
                if status == 'passed':
                    st.success("✅ 结构化校验通过（首次输出即通过）")
                elif status == 'repaired':
                    st.warning("⚠️ 原始输出未通过校验，自动修复后通过")
                elif status == 'failed':
                    st.error("❌ 结构化校验失败（自动修复后仍失败）")
                    with st.expander("查看失败原因", expanded=False):
                        for err in validation_status.get('errors', []):
                            st.markdown(f"- {err}")

            dedup_report = st.session_state.get('dedup_report')
            if dedup_report:
                hard_removed = dedup_report.get('hard_removed', [])
                semantic_warnings = dedup_report.get('semantic_warnings', [])
                if hard_removed:
                    st.success(f"🧹 已执行硬去重，移除 {len(hard_removed)} 条确定性重复")
                    with st.expander("查看硬去重明细", expanded=False):
                        for item in hard_removed:
                            st.markdown(f"- 保留 {item.get('kept_id')}，移除 {item.get('removed_id')}（{item.get('reason')}）")
                if semantic_warnings:
                    st.warning(f"🔎 检测到 {len(semantic_warnings)} 条语义近似告警（未自动删除）")
                    with st.expander("查看语义近似告警", expanded=False):
                        for item in semantic_warnings:
                            st.markdown(
                                f"- {item.get('case_a')} vs {item.get('case_b')}，相似度 {item.get('similarity')}（{item.get('reason')}）"
                            )
            
            if st.session_state['res_data']:
                df = pd.DataFrame(st.session_state['res_data'])
                module_list = df['module'].unique() if 'module' in df.columns else []
                st.caption(f"📊 当前共 **{len(df)}** 条用例 | 覆盖模块: {', '.join(module_list)}")
                
                # 新增 "⚖️ 智能评估" Tab
                tab_table, tab_json, tab_eval = st.tabs(["📊 表格视图", "🔍 源码/编辑", "⚖️ 智能评估"])
                
                with tab_table:
                    display_results(df, st.session_state['res_data'])
                
                with tab_json:
                    json_str_val = json.dumps(st.session_state['res_data'], indent=2, ensure_ascii=False)
                    edited_json_str = st.text_area("直接编辑 JSON", value=json_str_val, height=600)

                # ==================== 智能评估模块 (新增) ====================
                with tab_eval:
                    st.markdown("### 🕵️ 质量质检 & 智能对抗评估")
                    st.info("利用 AI 扮演 'QA 验收负责人'，基于 PRD 和 RAG 规范对当前生成的用例进行查漏补缺。")
                    
                    # 1. 选填：上传标准答案
                    golden_file = st.file_uploader("上传标准参考用例 (可选，作为对比标杆)", type=['json', 'txt', 'md'], help="如果有已存在的正确用例，上传后 AI 将进行对比分析")
                    golden_content = ""
                    if golden_file:
                        golden_content = golden_file.getvalue().decode('utf-8')[:10000] # 限制长度

                    # 2. 评估按钮
                    if st.button("⚖️ 开始全面评估", use_container_width=True):
                        if evaluator:
                            with st.spinner("QA 专家正在审查用例... (检查覆盖率、逻辑一致性、去重)"):
                                report = evaluator.evaluate_cases(
                                    selected_model,
                                    st.session_state.get('prd_context', '无详细PRD'),
                                    st.session_state['res_data'],
                                    rag_context=st.session_state.get('rag_context', ''),
                                    golden_cases_content=golden_content
                                )
                                st.session_state['eval_report'] = report
                                st.toast("评估完成！")
                        else:
                            st.error("评估器未初始化")

                    # 3. 结果渲染
                    if st.session_state['eval_report']:
                        report = st.session_state['eval_report']
                        
                        # 仪表盘
                        c_total, c_rule, c_llm, c_cov = st.columns(4)
                        c_total.metric("综合评分", f"{report.get('score', 0)} 分", delta=None)
                        c_rule.metric("规则评分", f"{report.get('rule_score', 0)} 分", delta=None)
                        c_llm.metric("LLM评分", f"{report.get('llm_score', 0)} 分", delta=None)
                        c_cov.metric("覆盖率", f"{report.get('coverage_rate', 0)}%", delta=None)

                        st.info(f"**总评**: {report.get('summary', '无')}")

                        if report.get('score_breakdown'):
                            breakdown = report.get('score_breakdown', {})
                            rw = int((breakdown.get('rule_weight', 0)) * 100)
                            lw = int((breakdown.get('llm_weight', 0)) * 100)
                            st.caption(f"评分权重：规则 {rw}% + LLM {lw}%")

                        total_reqs = report.get('total_requirements', 0)
                        covered_reqs = report.get('covered_requirements', 0)
                        st.caption(f"需求点覆盖：{covered_reqs}/{total_reqs}")

                        if report.get('coverage_matrix'):
                            with st.expander("🧭 查看覆盖率矩阵", expanded=False):
                                matrix_df = pd.DataFrame(report.get('coverage_matrix', []))
                                if not matrix_df.empty:
                                    matrix_df['covered'] = matrix_df['covered'].map(lambda x: "是" if x else "否")
                                    st.dataframe(
                                        matrix_df[[
                                            'requirement_id',
                                            'requirement',
                                            'covered',
                                            'matched_case_id',
                                            'matched_module',
                                            'match_score'
                                        ]],
                                        use_container_width=True,
                                        hide_index=True
                                    )

                        if report.get('uncovered_requirements'):
                            with st.expander("📌 未覆盖需求点", expanded=False):
                                for item in report.get('uncovered_requirements', []):
                                    st.markdown(
                                        f"- **{item.get('requirement_id')}** {item.get('requirement')} (匹配分: {item.get('best_score')})"
                                    )

                        if report.get('rule_checks'):
                            with st.expander("📏 查看规则评分明细", expanded=False):
                                for item in report.get('rule_checks', []):
                                    name = item.get('name', '未知检查项')
                                    score = item.get('score', 0)
                                    max_score = item.get('max_score', 0)
                                    detail = item.get('detail', '')
                                    status = item.get('status', 'warn')
                                    prefix = "✅" if status == "pass" else ("⚠️" if status == "warn" else "❌")
                                    st.markdown(f"{prefix} **{name}**：{score}/{max_score}，{detail}")

                        if report.get('rule_issues'):
                            with st.expander("🧩 查看规则问题列表", expanded=False):
                                for issue in report.get('rule_issues', []):
                                    st.markdown(f"- {issue}")

                        if report.get('semantic_duplicate_warnings'):
                            with st.expander("🔎 评估阶段语义近似告警", expanded=False):
                                for item in report.get('semantic_duplicate_warnings', []):
                                    st.markdown(
                                        f"- {item.get('case_a')} vs {item.get('case_b')}，相似度 {item.get('similarity')}（{item.get('reason')}）"
                                    )

                        st.divider()

                        # 详情列表
                        e1, e2 = st.columns(2)
                        with e1:
                            st.markdown("#### ⚠️ 发现问题")
                            if report.get('coverage_gap'):
                                st.error(f"**漏测风险 (Gap)**:\n" + "\n".join([f"- {i}" for i in report['coverage_gap']]))
                            else:
                                st.success("未发现明显覆盖率缺失")
                                
                            if report.get('logic_issues'):
                                st.warning(f"**逻辑/幻觉风险**:\n" + "\n".join([f"- {i['id']}: {i['issue']}" for i in report['logic_issues']]))
                            else:
                                st.success("逻辑一致性良好")
                        
                        with e2:
                            st.markdown("#### 💡 优化建议")
                            if report.get('duplicates'):
                                st.warning(f"**重复冗余**:\n" + "\n".join([f"- {i}" for i in report['duplicates']]))
                            else:
                                st.success("无重复用例")
                            
                            if report.get('suggestions'):
                                st.info(f"**改进方向**:\n" + "\n".join([f"- {i}" for i in report['suggestions']]))

                st.divider()
                # 归档按钮
                if st.button("💾 确认最终版并归档入库", type="primary", use_container_width=True):
                    if rag_engine:
                        try:
                            final_data = json.loads(edited_json_str) if 'edited_json_str' in locals() else st.session_state['res_data']
                            with st.spinner("归档中..."):
                                summary = generate_summary(api_key, str(final_data), model_name=selected_model)
                                rag_engine.add_history_case(st.session_state.get('prd_context', '对话生成的用例'), final_data, summary=summary)
                                st.success(f"已归档: {summary}")
                                st.balloons()
                        except Exception as e:
                            st.error(f"归档失败: {e}")
            else:
                st.info("👈 请在左侧上传 PRD 文档")

    # ==================== Tab 2: 知识库管理 ====================
    with tab_manage:
        st.header("🗂️ 知识库管理后台")
        
        with st.expander("➕ 上传新知识", expanded=False):
            kb_file = st.file_uploader("上传规范文档/历史资料", type=["txt", "md", "pdf", "jpg", "png"], key="kb_upload")
            if kb_file and st.button("上传并处理", key="kb_btn"):
                if rag_engine:
                    with st.spinner(f"正在智能解析..."):
                        kb_file.seek(0)
                        parsed_text = ""
                        if "text" in kb_file.type:
                            parsed_text = kb_file.getvalue().decode("utf-8")
                        else:
                            parsed_text = rag_engine.parse_file_content(kb_file, kb_file.type, model_name=selected_model)
                        
                        summary = generate_summary(api_key, parsed_text[:5000], model_name=selected_model)
                        kb_file.seek(0)
                        rag_engine.add_knowledge(kb_file, summary=summary, content_text=parsed_text, model_name=selected_model)
                        st.success(f"✅ 已存入！摘要：{summary}")
                        st.rerun()

        st.divider()
        col_kb, col_hist = st.columns(2)
        
        def render_doc_list(doc_type, title, icon):
            st.subheader(f"{icon} {title}")
            if rag_engine:
                docs = rag_engine.list_documents(doc_type)
                if docs:
                    df = pd.DataFrame(docs)
                    st.dataframe(df[["文件名/标题", "AI摘要", "录入时间", "ID"]], use_container_width=True, hide_index=True)
                    c1, c2 = st.columns([3, 1])
                    input_key = f"id_{doc_type}"
                    del_id = c1.text_input("输入 ID 进行操作", key=input_key, placeholder=f"粘贴 ID")
                    
                    if c2.button("🗑️ 删除", key=f"del_{doc_type}"):
                        if del_id:
                            rag_engine.delete_document(del_id, doc_type)
                            st.success(f"ID {del_id} 已删除"); st.rerun()
                            
                    if c2.button("👀 预览", key=f"view_{doc_type}"):
                        if del_id:
                            target = next((d for d in docs if d['ID'] == del_id), None)
                            if target:
                                st.info(f"正在预览: {target['文件名/标题']}")
                                content = rag_engine.get_doc_content(target['原始路径'], doc_id=target['ID'], collection_type=doc_type)
                                lang = "json" if doc_type == "history" else "markdown"
                                if lang == "json":
                                    try:
                                        content_obj = json.loads(content)
                                        content = json.dumps(content_obj, indent=2, ensure_ascii=False)
                                    except: pass
                                if "无法获取" in content: st.warning(content)
                                else: st.code(content, language=lang)
        with col_kb: render_doc_list("knowledge", "技术规范", "📚")
        with col_hist: render_doc_list("history", "历史案例", "🕰️")

    # ==================== Tab 3: 回归看板 ====================
    with tab_regression:
        st.header("📈 最小回归集与指标对比看板")
        st.caption("对比 Baseline 与 Upgraded 的结构化、去重、规则分、覆盖率等关键指标。")

        c_run, c_refresh = st.columns([1, 1])
        if c_run.button("▶️ 运行最小回归套件", use_container_width=True):
            with st.spinner("正在执行回归套件并生成对比报告..."):
                report = run_regression_suite()
                st.session_state['regression_report'] = report
                st.success("回归报告已生成")

        if c_refresh.button("🔄 读取最近一次报告", use_container_width=True):
            report = load_latest_regression_report()
            st.session_state['regression_report'] = report if report else None

        report = st.session_state.get('regression_report')
        if not report:
            report = load_latest_regression_report()
            if report:
                st.session_state['regression_report'] = report

        if not report:
            st.info("暂无回归报告，请先点击“运行最小回归套件”。")
        else:
            st.caption(f"报告时间：{report.get('generated_at', '-')}")
            st.caption(f"样本数：{report.get('suite_size', 0)}")

            baseline = report.get('baseline', {})
            upgraded = report.get('upgraded', {})
            delta = report.get('delta', {})

            st.markdown("### 📊 指标总览")
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Schema通过率",
                f"{upgraded.get('schema_pass_rate', 0)}%",
                f"{delta.get('schema_pass_rate', 0)}%"
            )
            m2.metric(
                "平均规则分",
                f"{upgraded.get('avg_rule_score', 0)}",
                f"{delta.get('avg_rule_score', 0)}"
            )
            m3.metric(
                "平均覆盖率",
                f"{upgraded.get('avg_coverage_rate', 0)}%",
                f"{delta.get('avg_coverage_rate', 0)}%"
            )

            m4, m5, m6 = st.columns(3)
            m4.metric(
                "硬去重移除总数",
                f"{upgraded.get('hard_duplicates_removed_total', 0)}",
                f"+{delta.get('hard_duplicates_improvement', 0)}"
            )
            m5.metric(
                "语义近似告警总数",
                f"{upgraded.get('semantic_warnings_total', 0)}",
                f"{delta.get('semantic_warnings_total', 0)}"
            )
            m6.metric(
                "修复样本数",
                f"{upgraded.get('repair_used_count', 0)}",
                delta=None
            )

            with st.expander("📋 Baseline vs Upgraded 明细", expanded=False):
                compare_rows = [
                    {
                        "指标": "解析成功率(%)",
                        "Baseline": baseline.get('parse_success_rate', 0),
                        "Upgraded": upgraded.get('parse_success_rate', 0),
                        "Delta": delta.get('parse_success_rate', 0)
                    },
                    {
                        "指标": "Schema通过率(%)",
                        "Baseline": baseline.get('schema_pass_rate', 0),
                        "Upgraded": upgraded.get('schema_pass_rate', 0),
                        "Delta": delta.get('schema_pass_rate', 0)
                    },
                    {
                        "指标": "平均规则分",
                        "Baseline": baseline.get('avg_rule_score', 0),
                        "Upgraded": upgraded.get('avg_rule_score', 0),
                        "Delta": delta.get('avg_rule_score', 0)
                    },
                    {
                        "指标": "平均覆盖率(%)",
                        "Baseline": baseline.get('avg_coverage_rate', 0),
                        "Upgraded": upgraded.get('avg_coverage_rate', 0),
                        "Delta": delta.get('avg_coverage_rate', 0)
                    },
                    {
                        "指标": "硬重复(基线识别) / 硬去重移除(升级后)",
                        "Baseline": baseline.get('hard_duplicates_total', 0),
                        "Upgraded": upgraded.get('hard_duplicates_removed_total', 0),
                        "Delta": delta.get('hard_duplicates_improvement', 0)
                    },
                    {
                        "指标": "语义近似告警总数",
                        "Baseline": baseline.get('semantic_warnings_total', 0),
                        "Upgraded": upgraded.get('semantic_warnings_total', 0),
                        "Delta": delta.get('semantic_warnings_total', 0)
                    }
                ]
                st.table(pd.DataFrame(compare_rows))

            samples = report.get('samples', [])
            if samples:
                st.markdown("### 🧪 样本级回归结果")
                sample_df = pd.DataFrame(samples)
                st.dataframe(
                    sample_df[[
                        'id',
                        'title',
                        'baseline_schema_pass',
                        'upgraded_schema_pass',
                        'baseline_rule_score',
                        'upgraded_rule_score',
                        'baseline_coverage_rate',
                        'upgraded_coverage_rate',
                        'hard_removed',
                        'semantic_warnings',
                        'repair_used'
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

    # ==================== Tab 4: RAG 评测看板 ====================
    with tab_rag_eval:
        st.header("🧪 RAG 离线评测看板")
        st.caption("评测集驱动：自动计算 Recall@K、误召回率、幻觉率，并支持历史报告追踪。")

        offline_eval_mode = st.checkbox(
            "断网离线模式（不调用 API，使用本地词法召回）",
            value=False,
            help="开启后无需 API Key。若样本未提供 offline_answer，将仅评估召回指标，不计算幻觉率。"
        )

        run_col, refresh_col, history_col = st.columns([1, 1, 1])

        if run_col.button("▶️ 运行 RAG 离线评测", use_container_width=True):
            if (not offline_eval_mode) and (not api_key):
                st.error("请先在左侧配置 API Key")
            else:
                with st.spinner("正在执行 RAG 离线评测并生成报告..."):
                    report = run_rag_offline_evaluation(
                        api_key=api_key if api_key else None,
                        model_name=selected_model,
                        offline_mode=offline_eval_mode
                    )
                    st.session_state['rag_eval_report'] = report
                    st.session_state['rag_eval_history'] = list_rag_eval_reports()
                    st.success("RAG 评测报告已生成")

        if refresh_col.button("🔄 读取最新报告", use_container_width=True):
            report = load_latest_rag_eval_report()
            st.session_state['rag_eval_report'] = report if report else None

        if history_col.button("🗂️ 刷新历史记录", use_container_width=True):
            st.session_state['rag_eval_history'] = list_rag_eval_reports()

        rag_report = st.session_state.get('rag_eval_report')
        if not rag_report:
            rag_report = load_latest_rag_eval_report()
            if rag_report:
                st.session_state['rag_eval_report'] = rag_report

        history_rows = st.session_state.get('rag_eval_history', [])
        if not history_rows:
            history_rows = list_rag_eval_reports()
            st.session_state['rag_eval_history'] = history_rows

        if history_rows:
            history_df = pd.DataFrame(history_rows)
            with st.expander("📚 历史报告", expanded=False):
                st.dataframe(
                    history_df[[
                        'file',
                        'generated_at',
                        'suite_name',
                        'suite_size',
                        'retrieval_mode',
                        'recall_at_k',
                        'false_recall_rate',
                        'hallucination_rate'
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

                file_options = history_df['file'].tolist()
                selected_file = st.selectbox("选择历史报告文件", file_options, key="rag_eval_history_file")
                if st.button("📂 加载所选历史报告", use_container_width=True):
                    selected_row = next((r for r in history_rows if r.get('file') == selected_file), None)
                    if selected_row:
                        loaded = load_rag_eval_report(selected_row.get('path', ''))
                        if loaded:
                            st.session_state['rag_eval_report'] = loaded
                            rag_report = loaded
                            st.success(f"已加载历史报告: {selected_file}")

        if not rag_report:
            st.info("暂无 RAG 评测报告，请先点击“运行 RAG 离线评测”。")
        else:
            st.caption(f"报告时间：{rag_report.get('generated_at', '-')}")
            st.caption(f"评测集：{rag_report.get('suite_name', '-')}")
            st.caption(f"样本数：{rag_report.get('suite_size', 0)}")
            st.caption(f"评测模式：{rag_report.get('retrieval_mode', '-')}")

            metrics = rag_report.get('metrics', {})
            m1, m2, m3 = st.columns(3)
            m1.metric("Recall@K", f"{metrics.get('recall_at_k', 0)}%")
            m2.metric("误召回率", f"{metrics.get('false_recall_rate', 0)}%")
            hall_rate = metrics.get('hallucination_rate', None)
            hall_rate_display = "-" if hall_rate is None else f"{hall_rate}%"
            m3.metric("幻觉率", hall_rate_display)

            m4, m5 = st.columns(2)
            m4.metric("幻觉样本数", f"{metrics.get('hallucinated_samples', 0)}")
            m5.metric("幻觉可评估样本", f"{metrics.get('hallucination_evaluable_samples', 0)}")

            sample_rows = rag_report.get('samples', [])
            if sample_rows:
                st.markdown("### 🧾 样本级评测结果")
                sample_df = pd.DataFrame(sample_rows)
                sample_df['recall_at_k_display'] = sample_df['recall_at_k'].apply(
                    lambda x: "-" if pd.isna(x) else f"{round(float(x) * 100, 1)}%"
                )
                sample_df['false_recall_rate_display'] = sample_df['false_recall_rate'].apply(
                    lambda x: f"{round(float(x) * 100, 1)}%"
                )
                sample_df['hallucinated_display'] = sample_df['hallucinated'].apply(
                    lambda x: "-" if pd.isna(x) else ("是" if bool(x) else "否")
                )
                sample_df['hit_sources_display'] = sample_df['hit_sources'].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) and x else "-"
                )
                sample_df['false_sources_display'] = sample_df['false_retrieved_sources'].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) and x else "-"
                )
                sample_df['hallucination_reason_display'] = sample_df['hallucination_reasons'].apply(
                    lambda x: " | ".join(x) if isinstance(x, list) and x else "-"
                )

                st.dataframe(
                    sample_df[[
                        'id',
                        'title',
                        'recall_at_k_display',
                        'false_recall_rate_display',
                        'hallucinated_display',
                        'hit_sources_display',
                        'false_sources_display',
                        'hallucination_reason_display'
                    ]],
                    use_container_width=True,
                    hide_index=True
                )

                with st.expander("🔍 查看样本回答详情", expanded=False):
                    for row in sample_rows:
                        st.markdown(f"**{row.get('id', '')} - {row.get('title', '')}**")
                        st.caption(f"问题: {row.get('query', '')}")
                        st.info(row.get('answer', ''))
                        st.caption(
                            f"召回来源: {', '.join(row.get('retrieved_sources', [])) if row.get('retrieved_sources') else '-'}"
                        )
                        st.divider()

if __name__ == "__main__":
    main()