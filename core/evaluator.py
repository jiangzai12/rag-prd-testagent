import google.generativeai as genai
import json
import sys
import os

# 路径适配
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.prompts import PromptManager
from core.llm_client import extract_json_from_text
from core.rule_scorer import score_test_cases_by_rules
from core.coverage_matrix import build_coverage_matrix


def _safe_llm_report(report_json, raw_text=""):
    """兼容并规范化 LLM 评审输出结构。"""
    if not isinstance(report_json, dict):
        return {
            "score": 0,
            "summary": "AI 未能生成有效的 JSON 格式报告，请重试。",
            "coverage_gap": [],
            "logic_issues": [],
            "duplicates": [],
            "suggestions": [f"原始响应: {raw_text[:200]}..."] if raw_text else ["原始响应为空"]
        }

    score = report_json.get("score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0

    return {
        "score": max(0, min(100, score)),
        "summary": str(report_json.get("summary", "AI 评审完成")),
        "coverage_gap": report_json.get("coverage_gap", []) if isinstance(report_json.get("coverage_gap", []), list) else [],
        "logic_issues": report_json.get("logic_issues", []) if isinstance(report_json.get("logic_issues", []), list) else [],
        "duplicates": report_json.get("duplicates", []) if isinstance(report_json.get("duplicates", []), list) else [],
        "suggestions": report_json.get("suggestions", []) if isinstance(report_json.get("suggestions", []), list) else []
    }

class Evaluator:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Evaluator 需要 API Key")
        self.api_key = api_key
        
    def evaluate_cases(self, model_name, prd_text, current_cases, rag_context=None, golden_cases_content=None):
        """
        执行测试用例评估
        
        Args:
            model_name: 使用的模型 (建议使用 Pro 版本以获得更好的逻辑推理能力)
            prd_text: 原始需求文本
            current_cases: 当前 AI 生成的测试用例 (List/Dict 或 JSON String)
            rag_context: RAG 检索到的规范上下文
            golden_cases_content: (可选) 人工上传的标准用例内容
            
        Returns:
            dict: 包含分数、建议等信息的结构化报告
        """
        rule_report = score_test_cases_by_rules(current_cases, prd_text=prd_text, rag_context=rag_context or "")
        coverage_report = build_coverage_matrix(prd_text or "", current_cases)

        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name, 
                system_instruction=PromptManager.EVALUATOR_SYSTEM_PROMPT
            )
            
            # 构建 Prompt
            prompt_text = PromptManager.get_evaluation_prompt(
                prd_text, 
                current_cases, 
                rag_text=rag_context, 
                golden_cases_text=golden_cases_content
            )
            
            # 调用模型
            # 评估任务通常不需要流式传输，一次性生成即可
            response = model.generate_content(prompt_text)
            
            # 解析结果
            report_json = extract_json_from_text(response.text)

            llm_report = _safe_llm_report(report_json, raw_text=response.text)

            rule_score = rule_report.get("rule_score", 0)
            llm_score = llm_report.get("score", 0)
            final_score = round(rule_score * 0.7 + llm_score * 0.3)

            combined_summary = (
                f"规则分 {rule_score}，LLM 分 {llm_score}，覆盖率 {coverage_report.get('coverage_rate', 0)}%。"
                f" 规则结论：{rule_report.get('rule_summary', '')}；"
                f"AI 结论：{llm_report.get('summary', '')}"
            )

            rule_issues = rule_report.get("rule_issues", [])
            merged_suggestions = llm_report.get("suggestions", [])[:]
            for issue in rule_issues[:5]:
                merged_suggestions.append(f"规则提示: {issue}")

            merged_duplicates = llm_report.get("duplicates", [])[:]
            for dup in rule_report.get("rule_duplicates", [])[:5]:
                merged_duplicates.append(f"规则识别: {dup}")

            semantic_dup_warnings = rule_report.get("semantic_duplicate_warnings", [])
            for warn in semantic_dup_warnings[:6]:
                merged_suggestions.append(
                    f"语义近似告警: {warn.get('case_a')} 与 {warn.get('case_b')} (相似度 {warn.get('similarity')})"
                )

            uncovered = coverage_report.get("uncovered_requirements", [])
            for item in uncovered[:5]:
                merged_suggestions.append(
                    f"覆盖补充: {item.get('requirement_id')} {item.get('requirement')}"
                )

            coverage_gap = llm_report.get("coverage_gap", [])[:]
            for item in uncovered[:6]:
                coverage_gap.append(
                    f"{item.get('requirement_id')}: {item.get('requirement')}"
                )

            return {
                "score": final_score,
                "summary": combined_summary,
                "rule_score": rule_score,
                "llm_score": llm_score,
                "coverage_rate": coverage_report.get("coverage_rate", 0.0),
                "total_requirements": coverage_report.get("total_requirements", 0),
                "covered_requirements": coverage_report.get("covered_requirements", 0),
                "coverage_matrix": coverage_report.get("coverage_matrix", []),
                "uncovered_requirements": coverage_report.get("uncovered_requirements", []),
                "score_breakdown": {
                    "rule_weight": 0.7, # 规则权重
                    "llm_weight": 0.3 # LLM 权重  两个权重可以根据实际情况调整，比如对于复杂逻辑评审可以适当增加 LLM 权重，对于结构化校验可以增加规则权重
                },
                "rule_checks": rule_report.get("rule_checks", []),
                "rule_issues": rule_issues,
                "semantic_duplicate_warnings": semantic_dup_warnings,
                "coverage_gap": coverage_gap,
                "logic_issues": llm_report.get("logic_issues", []),
                "duplicates": merged_duplicates,
                "suggestions": merged_suggestions
            }
            
        except Exception as e:
            print(f"评估过程出错: {e}")
            rule_score = rule_report.get("rule_score", 0)
            fallback_suggestions = ["请检查网络连接或 API Key 配额"]
            for issue in rule_report.get("rule_issues", [])[:5]:
                fallback_suggestions.append(f"规则提示: {issue}")
            return {
                "score": rule_score,
                "summary": f"LLM 评估失败，已返回规则评分。错误: {str(e)}",
                "rule_score": rule_score,
                "llm_score": 0,
                "coverage_rate": coverage_report.get("coverage_rate", 0.0),
                "total_requirements": coverage_report.get("total_requirements", 0),
                "covered_requirements": coverage_report.get("covered_requirements", 0),
                "coverage_matrix": coverage_report.get("coverage_matrix", []),
                "uncovered_requirements": coverage_report.get("uncovered_requirements", []),
                "score_breakdown": {
                    "rule_weight": 1.0,
                    "llm_weight": 0.0
                },
                "rule_checks": rule_report.get("rule_checks", []),
                "rule_issues": rule_report.get("rule_issues", []),
                "semantic_duplicate_warnings": rule_report.get("semantic_duplicate_warnings", []),
                "coverage_gap": [
                    f"{item.get('requirement_id')}: {item.get('requirement')}"
                    for item in coverage_report.get("uncovered_requirements", [])[:6]
                ],
                "logic_issues": [],
                "duplicates": [f"规则识别: {dup}" for dup in rule_report.get("rule_duplicates", [])[:5]],
                "suggestions": fallback_suggestions
            }