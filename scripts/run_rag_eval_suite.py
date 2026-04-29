import json
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import load_config
from core.rag_offline_eval import run_rag_offline_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG offline evaluation suite")
    parser.add_argument("--offline", action="store_true", help="Use fully offline lexical retrieval mode")
    parser.add_argument("--model", type=str, default="models/gemini-1.5-flash", help="Model name for online mode")
    args = parser.parse_args()

    config = load_config()
    api_key = config.get("api_key", "")
    if not args.offline and not api_key:
        raise ValueError("请先在 data/user_config.json 中配置 api_key")

    report = run_rag_offline_evaluation(
        api_key=api_key if api_key else None,
        model_name=args.model,
        offline_mode=args.offline
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
