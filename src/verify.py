"""Run the tests and record an actual API example in the result ledger."""

import json
import subprocess
import sys
import importlib.metadata
import xml.etree.ElementTree as ET
from fastapi.testclient import TestClient
from api.main import create_app
from src.data import ROOT, sha256
from src.evaluate import write_results


def main() -> None:
    report = ROOT / "reports/pytest.xml"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)
    suites = list(ET.parse(report).getroot().iter("testsuite"))
    totals = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    text = "Accountant managing audits, financial statements and tax reporting"
    with TestClient(create_app()) as client:
        response = client.post("/predict", json={"text": text})
        response.raise_for_status()
        health = client.get("/health")
        health.raise_for_status()
    results = json.loads((ROOT / "results.json").read_text())
    results["environment"]["versions"]["scipy"] = importlib.metadata.version("scipy")
    results["environment"]["requirements_sha256"] = sha256(ROOT / "requirements.txt")
    scores = results["models"][results["selected_model"]]
    matrix = scores["confusion_matrix"]
    scores["per_class_recall"] = {
        label: matrix[i][i] / sum(matrix[i])
        for i, label in enumerate(results["classes"])
    }
    results["verification"] = {
        "pytest": totals,
        "api_example": {
            "request": {"text": text},
            "response": response.json(),
            "status_code": response.status_code,
        },
        "health": health.json(),
    }
    write_results(results, ROOT / "results.json")
    print(json.dumps(results["verification"], indent=2))


if __name__ == "__main__":
    main()
