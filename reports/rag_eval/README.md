# RAG 离线评测报告目录

该目录用于存放 RAG 问答离线评测产物。

- latest_rag_eval_report.json: 机器可读指标报告。
- latest_rag_eval_report.md: 人类可读评测看板。
- history/: 每次运行生成一个时间戳快照，便于历史追踪。

## 生成方式

1. UI 方式: 打开应用后进入 "🧪 评测看板" Tab，点击 "▶️ 运行 RAG 离线评测"。
2. 命令行方式: python scripts/run_rag_eval_suite.py
3. 纯离线方式(断网可用): python scripts/run_rag_eval_suite.py --offline

## 断网模式说明

- 离线模式不调用 Embedding/LLM API，使用本地词法召回评估 Recall@K 与误召回率。
- 若 qa_samples 提供 offline_answer 字段，则会额外计算幻觉率。
- 若未提供 offline_answer，则该样本只评估召回指标，幻觉率记为不可评估。

## 指标说明

- Recall@K: 每个问题在 Top-K 召回中命中期望来源的平均比例。
- 误召回率: Top-K 召回来源中非期望来源的平均占比。
- 幻觉率: 答案命中禁用关键词、无依据强答或不可回答却未拒答的比例。
