import json
import os
from datetime import datetime
from typing import Any, Dict, List

from core.case_validator import validate_test_cases
from core.coverage_matrix import build_coverage_matrix
from core.dedup_engine import apply_dedup_strategy, find_hard_duplicate_pairs
from core.llm_client import extract_json_from_text, extract_json_with_error
from core.rule_scorer import score_test_cases_by_rules


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SUITE_PATH = os.path.join(BASE_DIR, "test_prd", "regression_suite.json")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "regression")
LATEST_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "latest_regression_report.json")
LATEST_MD = os.path.join(DEFAULT_OUTPUT_DIR, "latest_regression_report.md")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _safe_case_list(data: Any) -> List[Dict[str, Any]]:
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def _round_rate(value: float) -> float:
    return round(value, 1)


def _compute_metrics(case_list: List[Dict[str, Any]], prd_text: str) -> Dict[str, Any]:
    hard_pairs = find_hard_duplicate_pairs(case_list)
    rule = score_test_cases_by_rules(case_list, prd_text=prd_text)
    coverage = build_coverage_matrix(prd_text, case_list)
    is_schema_valid, schema_errors, _ = validate_test_cases(case_list)
    return {
        "case_count": len(case_list),
        "schema_pass": bool(is_schema_valid),
        "schema_error_count": 0 if is_schema_valid else len(schema_errors),
        "hard_duplicate_count": len(hard_pairs),
        "semantic_warning_count": len(rule.get("semantic_duplicate_warnings", [])),
        "rule_score": int(rule.get("rule_score", 0)),
        "coverage_rate": float(coverage.get("coverage_rate", 0.0))
    }


def _baseline_process(raw_response: str, prd_text: str) -> Dict[str, Any]:
    parsed = extract_json_from_text(raw_response)
    cases = _safe_case_list(parsed)
    metrics = _compute_metrics(cases, prd_text) if cases else {
        "case_count": 0,
        "schema_pass": False,
        "schema_error_count": 1,
        "hard_duplicate_count": 0,
        "semantic_warning_count": 0,
        "rule_score": 0,
        "coverage_rate": 0.0
    }
    metrics["parse_success"] = parsed is not None
    return metrics


def _upgraded_process(raw_response: str, prd_text: str, repaired_cases: Any = None) -> Dict[str, Any]:
    parsed, parse_error = extract_json_with_error(raw_response)
    parse_success = parsed is not None
    repair_used = False

    candidate_cases = _safe_case_list(parsed)
    is_valid = False

    if candidate_cases:
        is_valid, _, normalized = validate_test_cases(candidate_cases)
        if is_valid:
            candidate_cases = normalized

    if not is_valid and isinstance(repaired_cases, list):
        rep_valid, _, rep_norm = validate_test_cases(repaired_cases)
        if rep_valid:
            candidate_cases = rep_norm
            is_valid = True
            repair_used = True

    if not is_valid:
        return {
            "parse_success": parse_success,
            "parse_error": parse_error or "结构化校验失败",
            "repair_used": repair_used,
            "case_count": 0,
            "schema_pass": False,
            "schema_error_count": 1,
            "hard_duplicate_removed": 0,
            "semantic_warning_count": 0,
            "rule_score": 0,
            "coverage_rate": 0.0
        }

    dedup = apply_dedup_strategy(candidate_cases)
    final_cases = dedup.get("cases", [])
    metrics = _compute_metrics(final_cases, prd_text)
    metrics.update({
        "parse_success": parse_success,
        "parse_error": "",
        "repair_used": repair_used,
        "hard_duplicate_removed": len(dedup.get("hard_removed", [])),
        "semantic_warning_count": len(dedup.get("semantic_warnings", []))
    })
    return metrics


def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def _rate_true(flags: List[bool]) -> float:
    return _round_rate((sum(1 for f in flags if f) / len(flags)) * 100) if flags else 0.0


def _aggregate(samples: List[Dict[str, Any]], key: str) -> float:
    return _avg([float(sample.get(key, 0)) for sample in samples])


def _build_markdown_report(report: Dict[str, Any]) -> str:
    b = report["baseline"]
    u = report["upgraded"]
    d = report["delta"]

    lines = [
        "# 回归指标对比看板",
        "",
        f"- 生成时间: {report.get('generated_at', '')}",
        f"- 样本数量: {report.get('suite_size', 0)}",
        "",
        "## 总览",
        "",
        "| 指标 | Baseline | Upgraded | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| 解析成功率 | {b['parse_success_rate']}% | {u['parse_success_rate']}% | {d['parse_success_rate']}% |",
        f"| Schema通过率 | {b['schema_pass_rate']}% | {u['schema_pass_rate']}% | {d['schema_pass_rate']}% |",
        f"| 平均规则分 | {b['avg_rule_score']} | {u['avg_rule_score']} | {d['avg_rule_score']} |",
        f"| 平均覆盖率 | {b['avg_coverage_rate']}% | {u['avg_coverage_rate']}% | {d['avg_coverage_rate']}% |",
        f"| 硬重复总数/移除数 | {b['hard_duplicates_total']} | {u['hard_duplicates_removed_total']} | {d['hard_duplicates_improvement']} |",
        f"| 语义近似告警总数 | {b['semantic_warnings_total']} | {u['semantic_warnings_total']} | {d['semantic_warnings_total']} |",
        f"| 修复样本数 | 0 | {u['repair_used_count']} | {u['repair_used_count']} |",
        "",
        "## 样本明细",
        "",
        "| ID | 标题 | Baseline规则分 | Upgraded规则分 | Baseline覆盖率 | Upgraded覆盖率 | 硬去重移除 | 语义告警 | 修复 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    ]

    for sample in report.get("samples", []):
        lines.append(
            f"| {sample['id']} | {sample['title']} | {sample['baseline_rule_score']} | {sample['upgraded_rule_score']} | "
            f"{sample['baseline_coverage_rate']}% | {sample['upgraded_coverage_rate']}% | {sample['hard_removed']} | "
            f"{sample['semantic_warnings']} | {'是' if sample['repair_used'] else '否'} |"
        )

    return "\n".join(lines)


def run_regression_suite(suite_path: str = DEFAULT_SUITE_PATH, output_dir: str = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    suite = _load_json(suite_path)
    sample_rows: List[Dict[str, Any]] = []
    baseline_samples: List[Dict[str, Any]] = []
    upgraded_samples: List[Dict[str, Any]] = []

    for row in suite: 
        prd_path = _resolve_path(row.get("prd_path", ""))
        prd_text = _read_text(prd_path) if os.path.exists(prd_path) else ""
        raw_response = str(row.get("raw_response", ""))
        repaired_cases = row.get("repaired_cases")

        b = _baseline_process(raw_response, prd_text)
        u = _upgraded_process(raw_response, prd_text, repaired_cases=repaired_cases)

        baseline_samples.append(b)
        upgraded_samples.append(u)

        sample_rows.append({
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "baseline_parse_success": b.get("parse_success", False),
            "upgraded_parse_success": u.get("parse_success", False),
            "baseline_schema_pass": b.get("schema_pass", False),
            "upgraded_schema_pass": u.get("schema_pass", False),
            "baseline_rule_score": b.get("rule_score", 0),
            "upgraded_rule_score": u.get("rule_score", 0),
            "baseline_coverage_rate": b.get("coverage_rate", 0.0),
            "upgraded_coverage_rate": u.get("coverage_rate", 0.0),
            "hard_removed": u.get("hard_duplicate_removed", 0),
            "semantic_warnings": u.get("semantic_warning_count", 0),
            "repair_used": bool(u.get("repair_used", False))
        })

    baseline = {
        "parse_success_rate": _rate_true([s.get("parse_success", False) for s in baseline_samples]),
        "schema_pass_rate": _rate_true([s.get("schema_pass", False) for s in baseline_samples]),
        "avg_rule_score": _aggregate(baseline_samples, "rule_score"),
        "avg_coverage_rate": _aggregate(baseline_samples, "coverage_rate"),
        "hard_duplicates_total": int(sum(s.get("hard_duplicate_count", 0) for s in baseline_samples)),
        "semantic_warnings_total": int(sum(s.get("semantic_warning_count", 0) for s in baseline_samples))
    }

    upgraded = {
        "parse_success_rate": _rate_true([s.get("parse_success", False) for s in upgraded_samples]),
        "schema_pass_rate": _rate_true([s.get("schema_pass", False) for s in upgraded_samples]),
        "avg_rule_score": _aggregate(upgraded_samples, "rule_score"),
        "avg_coverage_rate": _aggregate(upgraded_samples, "coverage_rate"),
        "hard_duplicates_removed_total": int(sum(s.get("hard_duplicate_removed", 0) for s in upgraded_samples)),
        "semantic_warnings_total": int(sum(s.get("semantic_warning_count", 0) for s in upgraded_samples)),
        "repair_used_count": int(sum(1 for s in upgraded_samples if s.get("repair_used", False)))
    }

    delta = {
        "parse_success_rate": _round_rate(upgraded["parse_success_rate"] - baseline["parse_success_rate"]),
        "schema_pass_rate": _round_rate(upgraded["schema_pass_rate"] - baseline["schema_pass_rate"]),
        "avg_rule_score": _round_rate(upgraded["avg_rule_score"] - baseline["avg_rule_score"]),
        "avg_coverage_rate": _round_rate(upgraded["avg_coverage_rate"] - baseline["avg_coverage_rate"]),
        "hard_duplicates_improvement": int(upgraded["hard_duplicates_removed_total"]),
        "semantic_warnings_total": int(upgraded["semantic_warnings_total"] - baseline["semantic_warnings_total"])
    }

    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "suite_size": len(suite),
        "baseline": baseline,
        "upgraded": upgraded,
        "delta": delta,
        "samples": sample_rows
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(LATEST_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    markdown = _build_markdown_report(report)
    with open(LATEST_MD, "w", encoding="utf-8") as f:
        f.write(markdown)

    return report


def load_latest_regression_report(report_path: str = LATEST_JSON) -> Dict[str, Any]:
    if not os.path.exists(report_path):
        return {}
    return _load_json(report_path)
