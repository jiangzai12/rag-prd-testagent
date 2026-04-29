# RAG 离线评测看板

- 生成时间: 2026-03-15 13:42:24
- 评测集: login_security_rag_offline_suite_v1
- 样本数: 5
- 评测模型: models/gemini-1.5-flash
- 评测模式: offline_lexical

## 指标总览

| 指标 | 数值 |
| --- | ---: |
| Recall@K (平均) | 100.0% |
| 误召回率 (平均) | 60.0% |
| 幻觉率 | 0.0% |
| 幻觉样本数 | 0 |
| 幻觉可评估样本数 | 5 |

## 样本明细

| ID | 标题 | Recall@K | 误召回率 | 幻觉 | 命中来源 |
| --- | --- | ---: | ---: | ---: | --- |
| RAG_001 | 账号锁定阈值与时长 | 100.0% | 66.7% | 否 | login_lock_policy.md |
| RAG_002 | 会话失效机制 | 100.0% | 66.7% | 否 | session_timeout_policy.md |
| RAG_003 | 首次登录路由限制 | 100.0% | 66.7% | 否 | initial_password_policy.md |
| RAG_004 | 不可回答问题应拒答 | - | 100.0% | 否 | - |
| RAG_005 | P0 红线归纳 | 100.0% | 0.0% | 否 | login_lock_policy.md, initial_password_policy.md, session_timeout_policy.md |