import json

from core.regression_runner import run_regression_suite


if __name__ == "__main__":
    report = run_regression_suite()
    print(json.dumps(report, ensure_ascii=False, indent=2))
