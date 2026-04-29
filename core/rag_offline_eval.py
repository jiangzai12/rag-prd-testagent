import json
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import google.generativeai as genai

from core.rag_engine import DB_PATH, GeminiEmbeddingFunction, TextSplitter


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SUITE_PATH = os.path.join(BASE_DIR, "test_prd", "rag_eval_suite.json")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "reports", "rag_eval")
LATEST_JSON = os.path.join(DEFAULT_OUTPUT_DIR, "latest_rag_eval_report.json")
LATEST_MD = os.path.join(DEFAULT_OUTPUT_DIR, "latest_rag_eval_report.md")
HISTORY_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "history")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _round_rate(value: float) -> float:
    return round(value, 1)


def _avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _normalize_for_lexical(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokenize_cn_en(text: str) -> List[str]:
    raw = _normalize_for_lexical(text)
    if not raw:
        return []

    # 英文/数字按词切，中文按字切，组合后做去重统计时更稳健
    en_tokens = re.findall(r"[a-z0-9_]+", raw)
    cn_tokens = re.findall(r"[\u4e00-\u9fff]", raw)
    return en_tokens + cn_tokens


def _lexical_similarity(query: str, chunk: str) -> float:
    q_tokens = _tokenize_cn_en(query)
    c_tokens = _tokenize_cn_en(chunk)
    if not q_tokens or not c_tokens:
        return 0.0

    q_set = set(q_tokens)
    c_set = set(c_tokens)
    overlap = len(q_set & c_set)
    recall = overlap / len(q_set)
    precision = overlap / len(c_set)
    return round(recall * 0.7 + precision * 0.3, 4)


def _build_offline_lexical_index(
    corpus_docs: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in corpus_docs:
        source = str(row.get("source", "unknown_source"))
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        chunks = TextSplitter.recursive_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            rows.append({
                "source": source,
                "chunk_index": i,
                "chunk": chunk
            })
    return rows


def _build_temp_eval_collection(
    api_key: str,
    corpus_docs: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> Tuple[chromadb.PersistentClient, str, Any]:
    client = chromadb.PersistentClient(path=DB_PATH)
    coll_name = f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    embedding_fn = GeminiEmbeddingFunction(api_key)
    collection = client.get_or_create_collection(name=coll_name, embedding_function=embedding_fn)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for row in corpus_docs:
        source = str(row.get("source", "unknown_source"))
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        chunks = TextSplitter.recursive_split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            continue

        doc_id = row.get("doc_id") or uuid.uuid4().hex[:8]
        for idx, chunk in enumerate(chunks):
            ids.append(f"{doc_id}_chunk_{idx}")
            docs.append(chunk)
            metas.append({
                "source": source,
                "doc_id": doc_id,
                "chunk_index": idx
            })

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)

    return client, coll_name, collection


def _cleanup_temp_collection(client: chromadb.PersistentClient, coll_name: str) -> None:
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass


def _retrieve_top_context(collection: Any, query: str, top_k: int) -> Dict[str, Any]:
    result = collection.query(query_texts=[query], n_results=max(1, top_k))

    docs = result.get("documents", [[]])
    metas = result.get("metadatas", [[]])
    distances = result.get("distances", [[]])

    docs_list = docs[0] if docs else []
    metas_list = metas[0] if metas else []
    dist_list = distances[0] if distances else []

    ranked_chunks: List[Dict[str, Any]] = []
    for i, chunk in enumerate(docs_list):
        meta = metas_list[i] if i < len(metas_list) else {}
        distance = float(dist_list[i]) if i < len(dist_list) else 9999.0
        ranked_chunks.append({
            "source": meta.get("source", "unknown_source"),
            "distance": round(distance, 4),
            "chunk": chunk
        })

    ranked_chunks.sort(key=lambda x: x.get("distance", 9999.0))
    top_chunks = ranked_chunks[:top_k]

    unique_sources: List[str] = []
    for item in top_chunks:
        src = item.get("source", "unknown_source")
        if src not in unique_sources:
            unique_sources.append(src)

    return {
        "top_sources": unique_sources,
        "top_chunks": top_chunks
    }


def _retrieve_top_context_offline(lexical_index: List[Dict[str, Any]], query: str, top_k: int) -> Dict[str, Any]:
    ranked: List[Dict[str, Any]] = []
    for row in lexical_index:
        score = _lexical_similarity(query, str(row.get("chunk", "")))
        ranked.append({
            "source": row.get("source", "unknown_source"),
            "distance": round(1.0 - score, 4),
            "chunk": row.get("chunk", ""),
            "score": score
        })

    ranked.sort(key=lambda x: (x.get("distance", 1.0), -x.get("score", 0.0)))
    top_chunks = ranked[:max(1, top_k)]

    unique_sources: List[str] = []
    for item in top_chunks:
        src = item.get("source", "unknown_source")
        if src not in unique_sources:
            unique_sources.append(src)

    return {
        "top_sources": unique_sources,
        "top_chunks": top_chunks
    }


def _generate_rag_answer(api_key: str, model_name: str, query: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        [f"[来源: {item.get('source')}]\n{item.get('chunk', '')}" for item in context_chunks]
    )

    prompt = f"""
你是一个严格的问答助手。
请仅依据提供的上下文回答问题，不可使用外部知识。
如果上下文信息不足，必须明确回答“信息不足，无法确定”。

【问题】
{query}

【上下文】
{context_text if context_text else '（无）'}

请输出简洁答案。
"""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"模型调用失败: {str(e)}"


def _contains_any(text: str, keywords: List[str]) -> List[str]:
    lowered = text.lower()
    hits: List[str] = []
    for kw in keywords:
        if kw.lower() in lowered:
            hits.append(kw)
    return hits


def _estimate_hallucination(sample: Dict[str, Any], answer: str, has_context: bool) -> Dict[str, Any]:
    forbidden_keywords = sample.get("forbidden_answer_keywords", []) or []
    unanswerable = bool(sample.get("unanswerable", False))

    forbidden_hits = _contains_any(answer, [str(x) for x in forbidden_keywords])
    uncertain_signals = ["信息不足", "无法确定", "未知", "不确定", "缺少信息"]
    uncertain_hits = _contains_any(answer, uncertain_signals)

    hallucinated = False
    reasons: List[str] = []

    if forbidden_hits:
        hallucinated = True
        reasons.append(f"命中禁用答案关键词: {', '.join(forbidden_hits)}")

    if unanswerable and not uncertain_hits:
        hallucinated = True
        reasons.append("不可回答问题未明确拒答")

    if not has_context and not uncertain_hits:
        hallucinated = True
        reasons.append("无检索上下文但给出确定性结论")

    return {
        "hallucinated": hallucinated,
        "reasons": reasons,
        "forbidden_hits": forbidden_hits,
        "uncertain": bool(uncertain_hits)
    }


def _resolve_answer_for_mode(
    sample: Dict[str, Any],
    retrieved_chunks: List[Dict[str, Any]],
    offline_mode: bool,
    api_key: Optional[str],
    model_name: str
) -> Tuple[str, Optional[bool], List[str], List[str]]:
    if offline_mode:
        offline_answer = str(sample.get("offline_answer", "")).strip()
        if not offline_answer:
            # 离线模式不强制生成答案，允许只评估召回指标
            return "(离线模式未提供 offline_answer，仅完成检索评测)", None, ["离线模式未提供答案，未计算幻觉率"], []

        hall = _estimate_hallucination(sample, offline_answer, has_context=bool(retrieved_chunks))
        return offline_answer, hall.get("hallucinated"), hall.get("reasons", []), hall.get("forbidden_hits", [])

    answer = _generate_rag_answer(str(api_key or ""), model_name, str(sample.get("query", "")), retrieved_chunks)
    hall = _estimate_hallucination(sample, answer, has_context=bool(retrieved_chunks))
    return answer, hall.get("hallucinated"), hall.get("reasons", []), hall.get("forbidden_hits", [])


def _evaluate_one_sample(
    sample: Dict[str, Any],
    retriever: Any,
    model_name: str,
    offline_mode: bool,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    query = str(sample.get("query", "")).strip()
    top_k = int(sample.get("top_k", 3) or 3)

    if offline_mode:
        retrieval = _retrieve_top_context_offline(retriever, query=query, top_k=top_k)
    else:
        retrieval = _retrieve_top_context(retriever, query=query, top_k=top_k)
    retrieved_sources = retrieval.get("top_sources", [])
    retrieved_chunks = retrieval.get("top_chunks", [])

    expected_sources = sample.get("expected_sources", []) or []
    expected_sources = [str(s) for s in expected_sources]

    hit_sources = [src for src in retrieved_sources if src in expected_sources]
    false_sources = [src for src in retrieved_sources if src not in expected_sources]

    recall_at_k: Optional[float] = None
    if expected_sources:
        recall_at_k = round(len(set(hit_sources)) / len(set(expected_sources)), 4)

    false_recall_rate = round(
        (len(false_sources) / len(retrieved_sources)) if retrieved_sources else 0.0,
        4
    )

    answer, hallucinated, hall_reasons, forbidden_hits = _resolve_answer_for_mode(
        sample,
        retrieved_chunks,
        offline_mode=offline_mode,
        api_key=api_key,
        model_name=model_name
    )

    return {
        "id": sample.get("id", ""),
        "title": sample.get("title", ""),
        "query": query,
        "top_k": top_k,
        "expected_sources": expected_sources,
        "retrieved_sources": retrieved_sources,
        "hit_sources": hit_sources,
        "false_retrieved_sources": false_sources,
        "recall_at_k": recall_at_k,
        "false_recall_rate": false_recall_rate,
        "answer": answer,
        "hallucinated": hallucinated,
        "hallucination_reasons": hall_reasons,
        "forbidden_hits": forbidden_hits,
        "chunk_preview": [
            {
                "source": item.get("source"),
                "distance": item.get("distance"),
                "preview": str(item.get("chunk", ""))[:180]
            }
            for item in retrieved_chunks
        ]
    }


def _build_markdown_report(report: Dict[str, Any]) -> str:
    metrics = report.get("metrics", {})
    lines = [
        "# RAG 离线评测看板",
        "",
        f"- 生成时间: {report.get('generated_at', '')}",
        f"- 评测集: {report.get('suite_name', '')}",
        f"- 样本数: {report.get('suite_size', 0)}",
        f"- 评测模型: {report.get('model_name', '')}",
        f"- 评测模式: {report.get('retrieval_mode', '')}",
        "",
        "## 指标总览",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Recall@K (平均) | {metrics.get('recall_at_k', 0)}% |",
        f"| 误召回率 (平均) | {metrics.get('false_recall_rate', 0)}% |",
        f"| 幻觉率 | {metrics.get('hallucination_rate', '-') if metrics.get('hallucination_rate', None) is not None else '-'}% |" if metrics.get('hallucination_rate', None) is not None else "| 幻觉率 | - |",
        f"| 幻觉样本数 | {metrics.get('hallucinated_samples', 0)} |",
        f"| 幻觉可评估样本数 | {metrics.get('hallucination_evaluable_samples', 0)} |",
        "",
        "## 样本明细",
        "",
        "| ID | 标题 | Recall@K | 误召回率 | 幻觉 | 命中来源 |",
        "| --- | --- | ---: | ---: | ---: | --- |"
    ]

    for row in report.get("samples", []):
        recall = "-" if row.get("recall_at_k") is None else f"{round(float(row.get('recall_at_k', 0)) * 100, 1)}%"
        false_rate = f"{round(float(row.get('false_recall_rate', 0)) * 100, 1)}%"
        hall_value = row.get("hallucinated")
        hallucinated = "-" if hall_value is None else ("是" if hall_value else "否")
        hit_sources = ", ".join(row.get("hit_sources", [])) if row.get("hit_sources") else "-"
        lines.append(
            f"| {row.get('id', '')} | {row.get('title', '')} | {recall} | {false_rate} | {hallucinated} | {hit_sources} |"
        )

    return "\n".join(lines)


def run_rag_offline_evaluation(
    api_key: Optional[str] = None,
    model_name: str = "models/gemini-1.5-flash",
    suite_path: str = DEFAULT_SUITE_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    offline_mode: bool = False
) -> Dict[str, Any]:
    if not offline_mode and not api_key:
        raise ValueError("RAG 离线评测需要 API Key")

    suite = _load_json(suite_path)
    suite_name = str(suite.get("suite_name", "default_rag_eval_suite"))
    corpus = suite.get("corpus", []) or []
    samples = suite.get("qa_samples", []) or []

    if not corpus:
        raise ValueError("评测集缺少 corpus")
    if not samples:
        raise ValueError("评测集缺少 qa_samples")

    retrieval_mode = "offline_lexical" if offline_mode else "online_embedding_llm"

    client = None
    coll_name = ""
    retriever: Any = None
    if offline_mode:
        retriever = _build_offline_lexical_index(corpus)
    else:
        client, coll_name, retriever = _build_temp_eval_collection(str(api_key or ""), corpus)

    try:
        sample_reports: List[Dict[str, Any]] = []
        recall_values: List[float] = []
        false_recall_values: List[float] = []
        hall_values: List[bool] = []

        for sample in samples:
            sample_report = _evaluate_one_sample(
                sample,
                retriever=retriever,
                model_name=model_name,
                offline_mode=offline_mode,
                api_key=api_key
            )
            sample_reports.append(sample_report)

            if sample_report.get("recall_at_k") is not None:
                recall_values.append(float(sample_report.get("recall_at_k", 0.0)))
            false_recall_values.append(float(sample_report.get("false_recall_rate", 0.0)))
            if sample_report.get("hallucinated") is not None:
                hall_values.append(bool(sample_report.get("hallucinated")))

        hall_count = sum(1 for x in hall_values if x)
        hall_evaluable = len(hall_values)
        hall_rate = _round_rate((hall_count / hall_evaluable) * 100) if hall_evaluable else None

        metrics = {
            "recall_at_k": _round_rate(_avg(recall_values) * 100),
            "false_recall_rate": _round_rate(_avg(false_recall_values) * 100),
            "hallucination_rate": hall_rate,
            "hallucinated_samples": hall_count,
            "hallucination_evaluable_samples": hall_evaluable
        }

        now = datetime.now()
        report = {
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "suite_name": suite_name,
            "suite_size": len(samples),
            "model_name": model_name,
            "retrieval_mode": retrieval_mode,
            "metrics": metrics,
            "samples": sample_reports
        }

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(HISTORY_DIR, exist_ok=True)

        _dump_json(LATEST_JSON, report)
        with open(LATEST_MD, "w", encoding="utf-8") as f:
            f.write(_build_markdown_report(report))

        history_name = f"rag_eval_{now.strftime('%Y%m%d_%H%M%S')}.json"
        _dump_json(os.path.join(HISTORY_DIR, history_name), report)

        return report
    finally:
        if client and coll_name:
            _cleanup_temp_collection(client, coll_name)


def load_latest_rag_eval_report(report_path: str = LATEST_JSON) -> Dict[str, Any]:
    if not os.path.exists(report_path):
        return {}
    return _load_json(report_path)


def load_rag_eval_report(report_path: str) -> Dict[str, Any]:
    if not report_path or not os.path.exists(report_path):
        return {}
    return _load_json(report_path)


def list_rag_eval_reports(history_dir: str = HISTORY_DIR) -> List[Dict[str, Any]]:
    if not os.path.exists(history_dir):
        return []

    rows: List[Dict[str, Any]] = []
    for name in os.listdir(history_dir):
        if not name.endswith(".json"):
            continue

        path = os.path.join(history_dir, name)
        try:
            data = _load_json(path)
            metrics = data.get("metrics", {})
            rows.append({
                "file": name,
                "path": path,
                "generated_at": data.get("generated_at", ""),
                "suite_name": data.get("suite_name", ""),
                "suite_size": data.get("suite_size", 0),
                "retrieval_mode": data.get("retrieval_mode", ""),
                "recall_at_k": metrics.get("recall_at_k", 0.0),
                "false_recall_rate": metrics.get("false_recall_rate", 0.0),
                "hallucination_rate": metrics.get("hallucination_rate", 0.0)
            })
        except Exception:
            continue

    rows.sort(key=lambda x: x.get("file", ""), reverse=True)
    return rows
