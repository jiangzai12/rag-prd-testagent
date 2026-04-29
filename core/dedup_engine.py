import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip().lower()
    return re.sub(r"\s+", " ", text)


def _case_key(case: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    """确定性硬去重键：同业务语义字段完全一致才判定重复。"""
    return (
        _normalize_text(case.get("module", "")),
        _normalize_text(case.get("precondition", "")),
        _normalize_text(case.get("step", "")),
        _normalize_text(case.get("expected", "")),
        _normalize_text(case.get("priority", "")),
        _normalize_text(case.get("design_strategy", ""))
    )


def _case_text(case: Dict[str, Any]) -> str:
    return " ".join([
        _normalize_text(case.get("module", "")),
        _normalize_text(case.get("precondition", "")),
        _normalize_text(case.get("step", "")),
        _normalize_text(case.get("expected", "")),
        _normalize_text(case.get("design_strategy", ""))
    ]).strip()


def _char_ngrams(text: str, n: int = 2) -> set:
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(text_a: str, text_b: str) -> float: # 基于字符 n-gram 的 Jaccard 相似度，衡量文本的局部重叠程度
    set_a = _char_ngrams(text_a, n=2)
    set_b = _char_ngrams(text_b, n=2)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _hybrid_similarity(text_a: str, text_b: str) -> float: # 综合 Jaccard 和 SequenceMatcher，兼顾局部和全局相似度
    if not text_a or not text_b:
        return 0.0
    jac = _jaccard_similarity(text_a, text_b)
    seq = SequenceMatcher(None, text_a, text_b).ratio()
    return round(jac * 0.65 + seq * 0.35, 4)


def _extract_numbers(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"\d+(?:\.\d+)?", text)


def _to_case_list(cases: Any) -> List[Dict[str, Any]]:
    if isinstance(cases, list):
        return [item for item in cases if isinstance(item, dict)]
    if isinstance(cases, str):
        try:
            parsed = json.loads(cases)
            return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def find_hard_duplicate_pairs(cases: Any) -> List[Dict[str, Any]]:
    case_list = _to_case_list(cases)
    seen = {}
    duplicates: List[Dict[str, Any]] = []

    for idx, case in enumerate(case_list):
        key = _case_key(case)
        case_id = str(case.get("id", "")).strip() or f"IDX_{idx + 1}"
        if key in seen:
            kept = seen[key]
            duplicates.append({
                "kept_id": kept["id"],
                "removed_id": case_id,
                "reason": "业务字段完全一致（确定性重复）"
            })
        else:
            seen[key] = {"id": case_id, "index": idx}

    return duplicates


def hard_deduplicate_cases(cases: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """执行确定性硬去重：自动移除完全重复项，仅保留首条。"""
    case_list = _to_case_list(cases)
    deduped: List[Dict[str, Any]] = []
    seen = {}
    removed: List[Dict[str, Any]] = []

    for idx, case in enumerate(case_list):
        key = _case_key(case)
        case_id = str(case.get("id", "")).strip() or f"IDX_{idx + 1}"
        if key in seen:
            kept = seen[key]
            removed.append({
                "kept_id": kept["id"],
                "removed_id": case_id,
                "reason": "业务字段完全一致（确定性重复）"
            })
            continue

        seen[key] = {"id": case_id, "index": idx}
        deduped.append(case)

    return deduped, removed


def find_semantic_near_duplicates(
    cases: Any,
    threshold: float = 0.86,
    max_warnings: int = 20
) -> List[Dict[str, Any]]:
    """语义近似检测：仅给出告警，不做删除。"""
    case_list = _to_case_list(cases)
    warnings: List[Dict[str, Any]] = []

    for i in range(len(case_list)):
        case_a = case_list[i]
        text_a = _case_text(case_a)
        id_a = str(case_a.get("id", "")).strip() or f"IDX_{i + 1}"
        priority_a = _normalize_text(case_a.get("priority", ""))
        numbers_a = set(_extract_numbers(text_a))

        for j in range(i + 1, len(case_list)):
            case_b = case_list[j]
            text_b = _case_text(case_b)
            id_b = str(case_b.get("id", "")).strip() or f"IDX_{j + 1}"
            priority_b = _normalize_text(case_b.get("priority", ""))
            numbers_b = set(_extract_numbers(text_b))

            # 完全一致属于硬去重范围，这里不重复告警
            if _case_key(case_a) == _case_key(case_b):
                continue

            # 边界值保护：若数字集合都非空且不同，认为可能是不同边界场景，不告警
            if numbers_a and numbers_b and numbers_a != numbers_b:
                continue

            similarity = _hybrid_similarity(text_a, text_b)
            dynamic_threshold = threshold + 0.03 if priority_a != priority_b else threshold
            if similarity < dynamic_threshold:
                continue

            warnings.append({
                "case_a": id_a,
                "case_b": id_b,
                "similarity": round(similarity, 3),
                "reason": "语义高度相似，建议人工复核是否重复"
            })

            if len(warnings) >= max_warnings:
                return warnings

    return warnings


def apply_dedup_strategy(cases: Any) -> Dict[str, Any]:
    deduped_cases, hard_removed = hard_deduplicate_cases(cases)
    semantic_warnings = find_semantic_near_duplicates(deduped_cases)
    return {
        "cases": deduped_cases,
        "hard_removed": hard_removed,
        "semantic_warnings": semantic_warnings
    }
