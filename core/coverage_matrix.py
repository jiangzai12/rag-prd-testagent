import json
import re
from typing import Any, Dict, List, Tuple


ACTION_HINTS = [
    "必须", "应", "需要", "支持", "校验", "提示", "跳转", "锁定", "过期", "失败", "成功",
    "提交", "获取", "倒计时", "失效", "重置", "修改", "强制", "不能为空", "不一致", "错误"
]

WEAK_HINTS = [
    "登录", "密码", "验证码", "账号", "按钮", "输入框", "页面", "session", "token"
]

NOISE_HINTS = [
    "文档标题", "文档版本", "撰写日期", "文档状态", "文档介绍", "背景说明", "产品范围", "用户角色",
    "行业痛点", "未来展望", "核心技术栈", "demo", "附录"
]

LOW_VALUE_HINTS = [
    "本项目旨在", "通常", "说明书", "介绍", "场景", "范围"
]

PREFIX_PATTERN = r"^\s*((\d+(\.\d+){0,3}[\.、]?)|([a-zA-Z][\.、])|([一二三四五六七八九十]+[、\.]))\s*"


def _normalize_text(text: str) -> str: # 仅保留中文、英文和数字，去掉标点符号，并转换为小写
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text).lower()


def _char_ngrams(text: str, n: int = 2) -> set: 
    # 计算字符 n-gram 集合，n-gram 是指连续的 n 个字符组成的子串
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    tokens = re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]{2,8}", normalized)
    tokens = [t for t in tokens if len(t) >= 2]
    tokens.sort(key=len, reverse=True)

    dedup = []
    for token in tokens:
        if token not in dedup:
            dedup.append(token)
        if len(dedup) >= max_keywords:
            break
    return dedup


def _is_requirement_line(line: str) -> bool:
    if not line:
        return False

    stripped = line.strip()
    if len(stripped) < 6 or len(stripped) > 150:
        return False

    lowered = stripped.lower()
    if any(noise in lowered for noise in NOISE_HINTS):
        return False

    # 排除纯标题和分隔符
    if re.fullmatch(r"[=\-_*\s]+", stripped):
        return False
    if stripped.startswith("【") and stripped.endswith("】"):
        return False

    numbered = re.match(PREFIX_PATTERN, stripped)
    bullet = re.match(r"^\s*[-*•]\s*", stripped)
    has_action_hint = any(hint in stripped for hint in ACTION_HINTS)
    has_weak_hint = any(hint in lowered for hint in WEAK_HINTS)

    # 纯编号小标题通常很短，直接过滤
    if numbered:
        line_wo_num = re.sub(PREFIX_PATTERN, "", stripped)
        if len(line_wo_num) < 10 and not has_action_hint:
            return False

    return bool(has_action_hint or bullet or (numbered and has_weak_hint))


def _requirement_line_score(line: str) -> int:
    score = 0
    stripped = line.strip()
    lowered = stripped.lower()

    key_hints = ["必须", "应", "校验", "提示", "失败", "成功", "锁定", "过期", "强制", "错误", "不能为空", "不一致", "倒计时", "失效", "跳转"]
    for hint in key_hints:
        if hint in stripped:
            score += 4

    for hint in ACTION_HINTS:
        if hint in stripped:
            score += 2

    if re.match(PREFIX_PATTERN, stripped):
        score += 1
    if re.match(r"^\s*[-*•]\s*", stripped):
        score += 1

    for hint in LOW_VALUE_HINTS:
        if hint in lowered:
            score -= 3

    return score


def extract_requirement_points(prd_text: str, max_points: int = 20) -> List[Dict[str, Any]]:
    """从 PRD 文本中提取可用于覆盖率评估的需求点。"""
    if not isinstance(prd_text, str) or not prd_text.strip():
        return []

    candidates: List[Dict[str, Any]] = []
    seen = set()

    for idx, raw in enumerate(prd_text.splitlines(), start=1):
        line = raw.strip()
        if not _is_requirement_line(line):
            continue

        # 去掉常见序号前缀
        line = re.sub(PREFIX_PATTERN, "", line)
        line = re.sub(r"^\s*[-*•]\s*", "", line).strip()

        normalized = _normalize_text(line)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        candidates.append({
            "requirement": line,
            "source_line": idx,
            "keywords": _extract_keywords(line),
            "score": _requirement_line_score(line)
        })

    # 按评分优先，再按行号排序，优先保留更像“可测试需求”的句子
    candidates.sort(key=lambda x: (-x["score"], x["source_line"]))
    selected = candidates[:max_points]
    selected.sort(key=lambda x: x["source_line"])

    points: List[Dict[str, Any]] = []
    for i, item in enumerate(selected, start=1):
        points.append({
            "requirement_id": f"R{i:03d}",
            "requirement": item["requirement"],
            "source_line": item["source_line"],
            "keywords": item["keywords"]
        })

    # 兜底：如果未抽到需求点，取前几条有效文本
    if not points:
        fallback_lines = [
            line.strip() for line in prd_text.splitlines()
            if isinstance(line, str) and 8 <= len(line.strip()) <= 100 and not re.fullmatch(r"[=\-_*\s]+", line.strip())
        ][:8]
        for i, line in enumerate(fallback_lines, start=1):
            points.append({
                "requirement_id": f"R{i:03d}",
                "requirement": line,
                "source_line": -1,
                "keywords": _extract_keywords(line)
            })

    return points


def _to_case_list(current_cases: Any) -> List[Dict[str, Any]]:
    if isinstance(current_cases, list):
        return [c for c in current_cases if isinstance(c, dict)]
    if isinstance(current_cases, str):
        try:
            parsed = json.loads(current_cases)
            return [c for c in parsed if isinstance(c, dict)] if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _build_case_text(case: Dict[str, Any]) -> str:
    parts = [
        str(case.get("module", "")),
        str(case.get("precondition", "")),
        str(case.get("step", "")),
        str(case.get("expected", "")),
        str(case.get("design_strategy", ""))
    ]
    return " ".join(parts)


def _coverage_score(requirement_text: str, case_text: str, requirement_keywords: List[str]) -> float:
    # 计算需求与测试用例的覆盖度，结合字符 n-gram 重叠和关键词命中率
    req_norm = _normalize_text(requirement_text)
    case_norm = _normalize_text(case_text)
    if not req_norm or not case_norm:
        return 0.0

    req_grams = _char_ngrams(req_norm, n=2)
    case_grams = _char_ngrams(case_norm, n=2)
    if not req_grams:
        return 0.0

    gram_recall = len(req_grams & case_grams) / len(req_grams) 
    # 需求点的 n-gram 被测试用例覆盖的比例，作为覆盖度的主要衡量指标

    if requirement_keywords:
        hit_count = sum(1 for kw in requirement_keywords if kw and kw in case_norm)
        keyword_score = min(1.0, hit_count / max(1, len(requirement_keywords)))
    else:
        keyword_score = 0.0

    return round(gram_recall * 0.75 + keyword_score * 0.25, 4)


def build_coverage_matrix(prd_text: str, current_cases: Any, threshold: float = 0.22) -> Dict[str, Any]:
    """构建需求点覆盖率矩阵，返回覆盖率和未覆盖需求列表。"""
    requirements = extract_requirement_points(prd_text)
    cases = _to_case_list(current_cases)

    matrix: List[Dict[str, Any]] = []
    uncovered: List[Dict[str, Any]] = []

    for req in requirements:
        req_id = req["requirement_id"]
        req_text = req["requirement"]
        req_keywords = req.get("keywords", [])

        best_score = 0.0
        best_case_id = ""
        best_case_module = ""

        for case in cases:
            case_id = str(case.get("id", ""))
            case_module = str(case.get("module", ""))
            case_text = _build_case_text(case)
            score = _coverage_score(req_text, case_text, req_keywords)
            if score > best_score:
                best_score = score
                best_case_id = case_id
                best_case_module = case_module

        covered = best_score >= threshold
        row = {
            "requirement_id": req_id,
            "requirement": req_text,
            "covered": covered,
            "matched_case_id": best_case_id if covered else "",
            "matched_module": best_case_module if covered else "",
            "match_score": round(best_score, 3)
        }
        matrix.append(row)

        if not covered:
            uncovered.append({
                "requirement_id": req_id,
                "requirement": req_text,
                "best_score": round(best_score, 3)
            })

    total = len(matrix)
    covered_count = sum(1 for row in matrix if row["covered"])
    coverage_rate = round((covered_count / total) * 100, 1) if total else 0.0
    # 覆盖率保留一位小数，覆盖率计算方式：覆盖的需求点数量 / 总需求点数量 * 100%

    return {
        "coverage_rate": coverage_rate,
        "total_requirements": total,
        "covered_requirements": covered_count,
        "uncovered_requirements": uncovered,
        "coverage_matrix": matrix
    }
