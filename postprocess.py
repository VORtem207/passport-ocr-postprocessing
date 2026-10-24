"""Offline passport OCR post-processing: python postprocess.py input.jsonl answers.jsonl."""

import argparse
from datetime import date
import json
from pathlib import Path

from ocr_core import VALID_STATUSES, correct_fields, norm, quality_accepts, validity_status


# Source-specific thresholds: (long-field char_min, B field conf_min).
# Selected only on development; frozen before validation use number two.
CHAR_MIN = 0.999
THRESHOLDS = {"web": (0.995, 0.998), "mobile": (0.99, 0.99), "mobile_v2": (CHAR_MIN, 0.999)}
BOX_RATIO = 1.2
AS_OF = date(2026, 10, 1)


def answer_for(prediction):
    answer = {"doc_id": prediction["doc_id"], "decision": "reject"}
    thresholds = THRESHOLDS.get(prediction.get("source"))
    if thresholds is None:
        return answer
    fields = {f: norm(value) for f, value in correct_fields(prediction).items()}
    char_min, conf_min = thresholds
    if (quality_accepts(prediction, fields, char_min, BOX_RATIO, conf_min)
            and validity_status(fields["Birth_date"], fields["Issue_date"], AS_OF) in VALID_STATUSES):
        answer.update(decision="accept", fields=fields)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ")
    answers, seen = [], set()
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            prediction = json.loads(line)
            doc_id = prediction["doc_id"]
            if doc_id in seen:
                raise ValueError(f"Duplicate doc_id: {doc_id}")
            seen.add(doc_id)
            answers.append(answer_for(prediction))
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for answer in answers:
            handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
    accepted = sum(a["decision"] == "accept" for a in answers)
    print(f"Documents: {len(answers)}, accepted: {accepted}, rejected: {len(answers) - accepted}")


if __name__ == "__main__":
    main()
