# 回归报告目录

该目录用于存放最小回归套件的可复测产物。

- latest_regression_report.json: 机器可读指标报告。
- latest_regression_report.md: 人类可读看板报告。

## 生成方式

1. UI 方式: 打开应用后进入 "回归看板" Tab，点击 "运行最小回归套件"。
2. 命令行方式: `python scripts/run_regression_suite.py`。

## 指标说明

- Schema通过率: 用例结构校验通过的样本比例。
- 平均规则分: 规则评分器的平均分。
- 平均覆盖率: 需求点覆盖率矩阵平均值。
- 硬去重移除总数: 自动移除的确定性重复条数。
- 语义近似告警总数: 仅告警、不自动删除的近似重复条数。
