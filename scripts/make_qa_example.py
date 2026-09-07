#!/usr/bin/env python3
"""Write a one-line QA JSONL smoke fixture for evaluate_qa.py."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "data/qa_smoke.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "question": "In what year was the Royal Dutch Petroleum Company founded?",
        "passage": (
            "In February 1907, the Royal Dutch Shell Group was created through the "
            "amalgamation of two rival companies: the Royal Dutch Petroleum Company, "
            "founded in 1890, and the Shell Transport and Trading Company of the United Kingdom."
        ),
        "answers": ["1890"],
    }
    path.write_text(json.dumps(row) + "\n")
    print(path)


if __name__ == "__main__":
    main()
