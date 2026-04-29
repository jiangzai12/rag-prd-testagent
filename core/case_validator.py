REQUIRED_FIELDS = [
    "id",
    "module",
    "precondition",
    "step",
    "expected",
    "priority",
    "design_strategy"
]

ALLOWED_PRIORITIES = {"P0", "P1", "P2"}


def validate_test_cases(cases_data):
    """校验测试用例 JSON 结构与关键字段约束。"""
    if not isinstance(cases_data, list):
        return False, ["顶层数据必须是 JSON 数组(list)。"], None

    if len(cases_data) == 0:
        return False, ["测试用例数组不能为空。"], None

    errors = []
    normalized_cases = []
    seen_ids = {}

    for idx, item in enumerate(cases_data, start=1):
        item_prefix = f"第 {idx} 条"
        if not isinstance(item, dict):
            errors.append(f"{item_prefix} 不是 JSON 对象。")
            continue

        normalized = {}
        for key, value in item.items():
            normalized[key] = value.strip() if isinstance(value, str) else value

        for field in REQUIRED_FIELDS:
            if field not in normalized:
                errors.append(f"{item_prefix} 缺少必填字段: {field}")
            elif not isinstance(normalized[field], str):
                actual_type = type(normalized[field]).__name__
                errors.append(f"{item_prefix} 字段 {field} 必须是字符串，当前类型: {actual_type}")

        if isinstance(normalized.get("step"), str) and not normalized["step"]:
            errors.append(f"{item_prefix} 字段 step 不能为空。")

        if isinstance(normalized.get("expected"), str) and not normalized["expected"]:
            errors.append(f"{item_prefix} 字段 expected 不能为空。")

        priority = normalized.get("priority")
        if isinstance(priority, str) and priority not in ALLOWED_PRIORITIES:
            errors.append(f"{item_prefix} 字段 priority 只能是 P0/P1/P2，当前值: {priority}")

        case_id = normalized.get("id")
        if isinstance(case_id, str) and case_id:
            if case_id in seen_ids:
                errors.append(f"{item_prefix} 字段 id 重复: {case_id}（与第 {seen_ids[case_id]} 条重复）")
            else:
                seen_ids[case_id] = idx

        normalized_cases.append(normalized)

    if errors:
        return False, errors, None
    return True, [], normalized_cases
