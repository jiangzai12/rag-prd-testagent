import json
from typing import Any, Dict, List, Tuple

from core.case_validator import validate_test_cases
from core.dedup_engine import find_hard_duplicate_pairs, find_semantic_near_duplicates


BOUNDARY_HINT_WORDS = [
    "边界",
    "异常",
    "为空",
    "错误",
    "过期",
    "锁定",
    "无效",
    "非法",
    "超长",
    "超短"
]


def _to_case_list(current_cases: Any) -> List[Dict[str, Any]]:
    if isinstance(current_cases, list):
        return current_cases
    if isinstance(current_cases, str):
        try:
            parsed = json.loads(current_cases)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _contains_any(text: str, words: List[str]) -> bool:
    if not isinstance(text, str):
        return False
    return any(word in text for word in words)


def _normalize_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def score_test_cases_by_rules(current_cases: Any, prd_text: str = "", rag_context: str = "") -> Dict[str, Any]:
    """规则评分: 100 分满分，返回规则维度分数和可解释明细。"""
    del prd_text, rag_context

    cases = _to_case_list(current_cases)
    checks: List[Dict[str, Any]] = []
    issues: List[str] = []

    # 规则 1: 基础结构
    if cases:
        checks.append({
            "name": "基础结构有效",
            "score": 10,
            "max_score": 10,
            "status": "pass",
            "detail": "顶层为非空用例数组"
        })
    else:
        checks.append({
            "name": "基础结构有效",
            "score": 0,
            "max_score": 10,
            "status": "fail",
            "detail": "未解析到有效用例数组"
        })
        issues.append("未解析到有效测试用例数组")

    # 规则 2: Schema 字段完整性
    is_valid, validation_errors, normalized = validate_test_cases(cases)
    if is_valid:
        checks.append({
            "name": "字段完整性校验",
            "score": 25,
            "max_score": 25,
            "status": "pass",
            "detail": "必填字段、类型、priority、ID 唯一、空值校验均通过"
        })
    else:
        penalty = min(len(validation_errors), 5) * 5
        score = max(0, 25 - penalty)
        checks.append({
            "name": "字段完整性校验",
            "score": score,
            "max_score": 25,
            "status": "warn" if score > 0 else "fail",
            "detail": f"发现 {len(validation_errors)} 项结构问题"
        })
        issues.extend(validation_errors[:8])

    safe_cases = normalized if is_valid and normalized is not None else cases

    # 规则 3: 优先级分布
    priorities = [case.get("priority", "") for case in safe_cases if isinstance(case, dict)]
    if priorities and all(p in ["P0", "P1", "P2"] for p in priorities):
        has_p0 = any(p == "P0" for p in priorities)
        score = 15 if has_p0 else 10
        detail = "包含 P0 高优先级场景" if has_p0 else "未检测到 P0 场景"
        if not has_p0:
            issues.append("未检测到 P0 高优先级场景")
        checks.append({
            "name": "优先级合理性",
            "score": score,
            "max_score": 15,
            "status": "pass" if has_p0 else "warn",
            "detail": detail
        })
    else:
        checks.append({
            "name": "优先级合理性",
            "score": 0,
            "max_score": 15,
            "status": "fail",
            "detail": "priority 字段缺失或存在非法值"
        })
        issues.append("priority 字段缺失或存在非法值")

    # 规则 4: 重复检测（确定性硬去重 + 语义近似告警）
    hard_duplicate_pairs = find_hard_duplicate_pairs(safe_cases)
    semantic_warnings = find_semantic_near_duplicates(safe_cases)

    if hard_duplicate_pairs:
        deduction = min(len(hard_duplicate_pairs), 5) * 3
        checks.append({
            "name": "重复场景检测",
            "score": max(0, 15 - deduction),
            "max_score": 15,
            "status": "warn",
            "detail": f"发现 {len(hard_duplicate_pairs)} 组确定性重复"
        })
        issues.extend([
            f"硬重复: {pair.get('kept_id')} 与 {pair.get('removed_id')}"
            for pair in hard_duplicate_pairs[:5]
        ])
    else:
        checks.append({
            "name": "重复场景检测",
            "score": 15,
            "max_score": 15,
            "status": "pass",
            "detail": "未发现确定性重复"
        })

    if semantic_warnings:
        issues.extend([
            f"语义近似告警: {item.get('case_a')} 与 {item.get('case_b')} (相似度 {item.get('similarity')})"
            for item in semantic_warnings[:6]
        ])

    # 规则 5: 边界/异常覆盖提示
    hint_hits = 0
    for case in safe_cases:
        if not isinstance(case, dict):
            continue
        joined = " ".join([
            _normalize_text(case.get("step")),
            _normalize_text(case.get("expected")),
            _normalize_text(case.get("design_strategy"))
        ])
        if _contains_any(joined, BOUNDARY_HINT_WORDS):
            hint_hits += 1

    if safe_cases:
        ratio = hint_hits / max(1, len(safe_cases))
        if ratio >= 0.35:
            checks.append({
                "name": "边界与异常场景覆盖",
                "score": 20,
                "max_score": 20,
                "status": "pass",
                "detail": f"边界/异常场景占比 {ratio:.0%}"
            })
        elif ratio >= 0.15:
            checks.append({
                "name": "边界与异常场景覆盖",
                "score": 12,
                "max_score": 20,
                "status": "warn",
                "detail": f"边界/异常场景占比 {ratio:.0%}"
            })
            issues.append("边界/异常场景占比偏低，建议补充更多负向用例")
        else:
            checks.append({
                "name": "边界与异常场景覆盖",
                "score": 4,
                "max_score": 20,
                "status": "fail",
                "detail": f"边界/异常场景占比 {ratio:.0%}"
            })
            issues.append("边界/异常场景不足，可能导致漏测")
    else:
        checks.append({
            "name": "边界与异常场景覆盖",
            "score": 0,
            "max_score": 20,
            "status": "fail",
            "detail": "无可分析用例"
        })

    total_score = sum(item["score"] for item in checks)
    max_score = sum(item["max_score"] for item in checks)
    normalized_score = round((total_score / max_score) * 100) if max_score else 0

    if normalized_score >= 85:
        summary = "规则校验质量较高，结构化与覆盖性表现良好"
    elif normalized_score >= 70:
        summary = "规则校验基本通过，但仍存在可改进点"
    else:
        summary = "规则校验风险较高，建议优先修复结构化与覆盖问题"

    return {
        "rule_score": normalized_score,
        "rule_summary": summary,
        "rule_checks": checks,
        "rule_issues": issues,
        "rule_duplicates": [f"{pair.get('kept_id')} 与 {pair.get('removed_id')}" for pair in hard_duplicate_pairs],
        "semantic_duplicate_warnings": semantic_warnings
    }
