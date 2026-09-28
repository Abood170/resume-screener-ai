"""Capture real API responses to authored demo inputs; never print input text."""

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from api.main import create_app


def main(run: Path):
    # Authored illustrative inputs, not dataset records or performance evidence.
    cases = {
        "clear_accounting_demo": (
            "Accountant responsible for preparing financial statements and monthly accounting reports. "
            "Managed general ledger reconciliations, accounts payable, accounts receivable and payroll accounting. "
            "Prepared tax returns, supported internal audits and maintained accurate financial records. "
            "Analyzed budget variances, reviewed expense reports and reconciled bank statements. "
            "Used accounting software and spreadsheets to prepare balance sheets and income statements. "
            "Worked with the finance department on month end closing, regulatory compliance, audit documentation "
            "and financial reporting. Education includes accounting and financial management."
        ),
        "vague_short_demo": "Experienced professional with skills and teamwork",
    }
    responses = {}
    with TestClient(create_app()) as client:
        for name, text in cases.items():
            response = client.post("/predict", json={"text": text})
            response.raise_for_status()
            responses[name] = {
                "status_code": response.status_code,
                "response": response.json(),
            }
    with (run / "api_examples.json").open("x", encoding="utf-8") as output:
        json.dump(responses, output, indent=2, allow_nan=False)
    print(json.dumps(responses, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    main(parser.parse_args().run_dir.resolve())
